from contextlib import nullcontext
from typing import Optional, Tuple
from dataset import DisentanglementDataset
from evaluation.generator import LatentGenerator
from evaluation.metrics import (
    write_bvae_score,
    write_dci_score,
    write_mig_score,
    write_mod_expl_score,
)
from models import SoftIntroVAE
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ops import kl_divergence, reconstruction_loss


class VAESolver:
    def __init__(
        self,
        dataset: DisentanglementDataset,
        model: SoftIntroVAE,
        batch_size: int,
        optimizer_e: Optimizer,
        optimizer_d: Optimizer,
        beta_kl: float,
        beta_rec: float,
        device: torch.device,
        use_amp: bool,
        grad_scaler: Optional[GradScaler],
        writer: Optional[SummaryWriter] = None,
        test_iter: int = 1000,
    ):
        self.dataset = dataset
        if isinstance(self.dataset, DisentanglementDataset):
            self.latent_generator = LatentGenerator(self.dataset, device)
        self.model = model
        self.batch_size = batch_size
        self.optimizer_e = optimizer_e
        self.optimizer_d = optimizer_d
        self.beta_kl = beta_kl
        self.beta_rec = beta_rec
        self.device = device
        self.use_amp = use_amp
        self.grad_scaler = grad_scaler
        self.writer = writer
        self.test_iter = test_iter

        self.recon_loss_type = "mse"

    def train_step(self, batch: Tensor, cur_iter: int) -> None:
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real_batch = batch.to(self.device)

        # =========== Update E, D ================
        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            real_mu, real_logvar, z, rec = self.model(real_batch)

            loss_rec = reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )
            loss_kl = kl_divergence(real_logvar, real_mu, reduce="mean")

            loss = self.beta_rec * loss_rec + self.beta_kl * loss_kl

        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()

        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer_e)
            self.grad_scaler.step(self.optimizer_d)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_d.step()

        self.write_scalars(
            cur_iter,
            losses=dict(
                loss_rec=loss_rec.data.cpu().item(), loss_kl=loss_kl.data.cpu().item()
            ),
        )
        self._write_images_helper(real_batch, cur_iter)
        self.write_disentanglemnt_scores(cur_iter)

    def _write_images_helper(self, batch, cur_iter):
        if self.writer is not None and cur_iter % self.test_iter == 0:
            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, self.model.zdim)).to(self.device)
            fake = self.model.sample(noise_batch).to(self.device)
            self.write_images(batch, fake, cur_iter)

    def write_images(self, batch, fake_batch, cur_iter):
        if self.writer is not None and cur_iter % self.test_iter == 0:
            with torch.no_grad():
                _, _, _, rec_det = self.model(batch, deterministic=True)
                max_imgs = min(batch.size(0), 16)
                self.writer.add_images(
                    f"image_{cur_iter}",
                    torch.cat(
                        [
                            batch[:max_imgs],
                            rec_det[:max_imgs],
                            fake_batch[:max_imgs],
                        ],
                        dim=0,
                    ).data.cpu(),
                )

    def write_scalars(self, cur_iter: int, losses: dict, **kwargs):
        if self.writer is not None:
            self.write_losses(cur_iter, losses)
            for name, value in kwargs.items():
                self.writer.add_scalar(name, value, global_step=cur_iter)

    def write_losses(self, cur_iter: int, losses: dict):
        if self.writer is not None:
            self.writer.add_scalars("losses", losses, global_step=cur_iter)

    def write_disentanglemnt_scores(self, cur_iter: int, num_samples: int = 10000):
        if (
            self.writer is not None
            and isinstance(self.dataset, DisentanglementDataset)
            and cur_iter % self.test_iter == 0
        ):
            score_kwargs = dict(
                latent_generator=self.latent_generator,
                model=self.model,
                num_samples=num_samples,
                batch_size=self.batch_size,
            )
            print("Calculating disentanglment scores...")
            write_bvae_score(self.writer, cur_iter, **score_kwargs)
            # write_dci_score(self.writer, cur_iter, **score_kwargs)
            write_mig_score(self.writer, cur_iter, **score_kwargs)
            write_mod_expl_score(self.writer, cur_iter, **score_kwargs)
            print("Finished calculating disentanglemnt scores!")
