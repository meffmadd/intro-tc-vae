from dataset import DisentanglementDataset
from solvers.vae import VAESolver
from typing import Optional
from models import SoftIntroVAE
from ops import kl_divergence, total_correlation, reconstruction_loss

from contextlib import nullcontext
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

class TCSovler(VAESolver):
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
    
    def compute_kl_loss(self, z: Optional[Tensor], mu: Tensor, logvar: Tensor, reduce: str = "mean") -> Tensor:
        kl_loss = kl_divergence(logvar, mu, reduce=reduce)
        tc = (self.beta_kl - 1.) * total_correlation(z, mu, logvar, reduce=reduce)
        return tc + kl_loss
    
    def train_step(self, batch: Tensor, cur_iter: int) -> dict:
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real_batch = batch.to(self.device)

        # =========== Update E, D ================
        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            real_mu, real_logvar, z, rec = self.model(real_batch)

            loss_rec = reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )

            loss_kl = self.compute_kl_loss(z, real_mu, real_logvar)
            loss = self.beta_rec * loss_rec + self.beta_kl * loss_kl

        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()

        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
            if self.clip:
                self.grad_scaler.unscale_(self.optimizer_e)
                self.grad_scaler.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.grad_scaler.step(self.optimizer_e)
            self.grad_scaler.step(self.optimizer_d)
            self.grad_scaler.update()
        else:
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer_e.step()
            self.optimizer_d.step()

        if torch.isnan(loss):
            raise RuntimeError

        if self.writer:
            self.write_scalars(
                cur_iter,
                losses=dict(
                    loss_rec=self.beta_rec * loss_rec.data.cpu().item(), loss_kl=self.beta_kl * loss_kl.data.cpu().item()
                ),
            )
            self.writer.add_scalars(
                "losses_unscaled",
                dict(
                    r_loss=loss_rec.data.cpu().item(),
                    kl=loss_kl.data.cpu().item(),
                    expelbo_f=0,
                ),
                global_step=cur_iter,
            )
            self.write_gradient_norm(cur_iter)
            self._write_images_helper(real_batch, cur_iter)
            self.write_disentanglemnt_scores(cur_iter)
            self.writer.flush()
        
        return {
            "loss_enc": loss.data.cpu().item(),
            "loss_dec": loss.data.cpu().item(),
            "loss_kl": loss_kl.data.cpu().item(),
            "loss_rec": loss_rec.data.cpu().item(),
        }