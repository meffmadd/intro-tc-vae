from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
from dataset import DisentanglementDataset
from models import SoftIntroVAE
from solvers.vae import VAESolver
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ops import kl_divergence, reconstruction_loss, reparameterize


class IntroSolver(VAESolver):
    def __init__(
        self,
        dataset: DisentanglementDataset,
        model: SoftIntroVAE,
        batch_size: int,
        optimizer_e: Optimizer,
        optimizer_d: Optimizer,
        recon_loss_type: str,
        beta_kl: float,
        beta_rec: float,
        beta_neg: float,
        gamma_r: float,
        device: torch.device,
        use_amp: bool,
        grad_scaler: Optional[GradScaler],
        writer: Optional[SummaryWriter] = None,
        test_iter: int = 1000,
        clip: Optional[float] = None,
    ):
        super().__init__(
            dataset,
            model,
            batch_size,
            optimizer_e,
            optimizer_d,
            recon_loss_type,
            beta_kl,
            beta_rec,
            device,
            use_amp,
            grad_scaler,
            writer,
            test_iter,
            clip
        )
        self.beta_neg = beta_neg
        self.gamma_r = gamma_r
        # normalize by images size (channels * height * width)
        self.scale = 1 / (self.model.cdim * self.model.encoder.image_size**2)

    def train_step(self, batch: Tensor, cur_iter: int) -> Tuple[float, float]:
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        # =========== Update E ================
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, self.model.zdim)).to(self.device)

            real_batch = batch.to(self.device)

            fake = self.model.sample(noise_batch)
            real_mu, real_logvar, z, rec = self.model(real_batch)

            loss_rec = reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )
            lossE_real_kl = self.compute_kl_loss(z, real_mu, real_logvar)

            rec_mu, rec_logvar, z_rec, rec_rec = self.model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = self.model(fake.detach())

            kl_rec = self.compute_kl_loss(z, rec_mu, rec_logvar, reduce="none") # shape: (batch_size,)
            kl_fake = self.compute_kl_loss(z, fake_mu, fake_logvar, reduce="none") # shape: (batch_size,)

            if self.writer:
                self.writer.add_scalars(
                    "fake_mu",
                    {"min": fake_mu.min().item(), "max": fake_mu.max().item()},
                    global_step=cur_iter,
                )
                self.writer.add_scalars(
                    "fake_logvar",
                    {"min": fake_logvar.min().item(), "max": fake_logvar.max().item()},
                    global_step=cur_iter,
                )
                self.writer.add_scalars(
                    "kl_fake",
                    {"min": kl_fake.min().item(), "max": kl_fake.max().item()},
                    global_step=cur_iter,
                )
                self.writer.add_scalars(
                    "lossE_real_kl",
                    {"min": lossE_real_kl.min().item(), "max": lossE_real_kl.max().item()},
                    global_step=cur_iter,
                )
                self.writer.flush()

            loss_rec_rec_e = reconstruction_loss(
                rec, rec_rec, loss_type=self.recon_loss_type, reduction="none"
            ) # shape: (batch_size,)
            while len(loss_rec_rec_e.shape) > 1:
                loss_rec_rec_e = loss_rec_rec_e.sum(-1)
            loss_rec_fake_e = reconstruction_loss(
                fake, rec_fake, loss_type=self.recon_loss_type, reduction="none"
            ) # shape: (batch_size,)
            while len(loss_rec_fake_e.shape) > 1:
                loss_rec_fake_e = loss_rec_fake_e.sum(-1)

            expelbo_rec = (
                (
                    -2
                    * self.scale
                    * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)
                )
                .exp()
                .mean()
            )
            expelbo_fake = (
                (
                    -2
                    * self.scale
                    * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)
                )
                .exp()
                .mean()
            )

            lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
            lossE_real = self.scale * (
                self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl
            )

            lossE = lossE_real + lossE_fake

        self.optimizer_e.zero_grad()

        if self.use_amp:
            self.grad_scaler.scale(lossE).backward()
            if self.clip:
                self.grad_scaler.unscale_(self.optimizer_e)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.grad_scaler.step(self.optimizer_e)
            self.grad_scaler.update()
        else:
            lossE.backward()

            if self.writer:
                self.writer.add_scalar("encoder_max_grad", torch.cat([torch.abs(p.grad).view(-1) for p in self.model.parameters() if p.grad is not None]).max(), cur_iter)
                self.writer.flush()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer_e.step()

        # ========= Update D ==================
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            fake = self.model.sample(noise_batch)
            rec = self.model.decoder(z.detach())
            loss_rec = reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )

            rec_mu, rec_logvar = self.model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = self.model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = self.model.decode(z_rec.detach())
            rec_fake = self.model.decode(z_fake.detach())

            loss_rec_rec = reconstruction_loss(
                rec.detach(),
                rec_rec,
                loss_type=self.recon_loss_type,
                reduction="mean",
            )
            loss_fake_rec = reconstruction_loss(
                fake.detach(),
                rec_fake,
                loss_type=self.recon_loss_type,
                reduction="mean",
            )

            lossD_rec_kl = self.compute_kl_loss(z_rec, rec_mu, rec_logvar)
            lossD_fake_kl = self.compute_kl_loss(z_fake, fake_mu, fake_logvar)

            lossD = self.scale * (
                loss_rec * self.beta_rec
                + (lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl
                + self.gamma_r * 0.5 * self.beta_rec * (loss_rec_rec + loss_fake_rec)
            )

        self.optimizer_d.zero_grad()

        if self.use_amp:
            self.grad_scaler.scale(lossD).backward()
            if self.clip:
                self.grad_scaler.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.grad_scaler.step(self.optimizer_d)
            self.grad_scaler.update()
        else:
            lossD.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer_d.step()

        if torch.isnan(lossD) or torch.isnan(lossE):
            raise RuntimeError

        if self.writer:
            dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
            self.write_scalars(
                cur_iter,
                losses=dict(
                    r_loss=self.beta_rec * loss_rec.data.cpu().item(),
                    kl=self.beta_kl * lossE_real_kl.data.cpu().item(),
                    expelbo_f=expelbo_fake.cpu().item(),
                ),
                diff_kl=dif_kl.item(),
            )
            self.write_gradient_flow(cur_iter, self.model.named_parameters())
            self.writer.add_scalar("lossE", lossE, global_step=cur_iter)
            self.writer.add_scalar("lossD", lossD, global_step=cur_iter)
            self.write_gradient_norm(cur_iter)
            self.write_images(real_batch, fake, cur_iter)
            self.write_disentanglemnt_scores(cur_iter)
            self.writer.flush()
        
        return lossE, lossD
