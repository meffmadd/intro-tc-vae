from contextlib import nullcontext
from typing import Optional, Tuple
from models import SoftIntroVAE
from solvers.vae import VAESolver
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils import calc_kl, calc_reconstruction_loss, reparameterize


class IntroSolver(VAESolver):
    def __init__(
        self,
        model: SoftIntroVAE,
        optimizer_e: Optimizer,
        optimizer_d: Optimizer,
        beta_kl: float,
        beta_rec: float,
        beta_neg: float,
        device: torch.device,
        use_amp: bool,
        grad_scaler: Optional[GradScaler],
        writer: Optional[SummaryWriter] = None,
    ):
        super().__init__(
            model,
            optimizer_e,
            optimizer_d,
            beta_kl,
            beta_rec,
            device,
            use_amp,
            grad_scaler,
            writer,
        )
        self.beta_neg = beta_neg
        self.gamma_r = 1e-8
        # normalize by images size (channels * height * width)
        self.scale = 1 / self.model.cdim * self.model.encoder.image_size**2

    def train_step(self, batch: Tensor, cur_iter: int) -> None:
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        b_size = batch.size(0)
        noise_batch = torch.randn(size=(b_size, self.model.zdim)).to(self.device)

        real_batch = batch.to(self.device)

        # =========== Update E ================
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            fake = self.model.sample(noise_batch)

            real_mu, real_logvar = self.model.encode(real_batch)
            z = reparameterize(real_mu, real_logvar)
            rec = self.model.decoder(z)

            loss_rec = calc_reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )

            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

            rec_mu, rec_logvar, z_rec, rec_rec = self.model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = self.model(fake.detach())

            kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
            kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

            loss_rec_rec_e = calc_reconstruction_loss(
                rec, rec_rec, loss_type=self.recon_loss_type, reduction="none"
            )
            while len(loss_rec_rec_e.shape) > 1:
                loss_rec_rec_e = loss_rec_rec_e.sum(-1)
            loss_rec_fake_e = calc_reconstruction_loss(
                fake, rec_fake, loss_type=self.recon_loss_type, reduction="none"
            )
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
            self.grad_scaler.step(self.optimizer_e)
        else:
            lossE.backward()
            self.optimizer_e.step()

        # ========= Update D ==================
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            fake = self.model.sample(noise_batch)
            rec = self.model.decoder(z.detach())
            loss_rec = calc_reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )

            rec_mu, rec_logvar = self.model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = self.model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = self.model.decode(z_rec.detach())
            rec_fake = self.model.decode(z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(
                rec.detach(),
                rec_rec,
                loss_type=self.recon_loss_type,
                reduction="mean",
            )
            loss_fake_rec = calc_reconstruction_loss(
                fake.detach(),
                rec_fake,
                loss_type=self.recon_loss_type,
                reduction="mean",
            )

            lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
            lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

            lossD = self.scale * (
                loss_rec * self.beta_rec
                + (lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl
                + self.gamma_r * 0.5 * self.beta_rec * (loss_rec_rec + loss_fake_rec)
            )

        self.optimizer_d.zero_grad()

        if self.use_amp:
            self.grad_scaler.scale(lossD).backward()
            self.grad_scaler.step(self.optimizer_d)
            self.grad_scaler.update()
        else:
            lossD.backward()
            self.optimizer_d.step()

        if torch.isnan(lossD) or torch.isnan(lossE):
            raise SystemError

        dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
        self.write_scalars(
            cur_iter,
            losses=dict(
                r_loss=loss_rec.data.cpu().item(),
                kl=lossE_real_kl.data.cpu().item(),
                expelbo_f=expelbo_fake.cpu().item(),
            ),
            diff_kl=dif_kl.item(),
        )
        self.write_images(batch, fake, cur_iter)
