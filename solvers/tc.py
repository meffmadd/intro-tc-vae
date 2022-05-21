from dataset import DisentanglementDataset
from solvers.vae import VAESolver
from typing import Optional
from models import SoftIntroVAE
from ops import log_qz_cond_x, log_pz, log_prod_qz_i, log_qz, reconstruction_loss

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
            # instead of loss_kl we take the decomposed term (Equation 2)
            logqz_condx = log_qz_cond_x(z, mu, logvar)
            logpz = log_pz(z)
            # with minibatch weighted sampling:
            logqz_prodmarginals = log_prod_qz_i(z, mu, logvar)
            logqz = log_qz(z, mu, logvar)

            mi_loss = logqz_condx - logqz
            tc_loss = logqz - logqz_prodmarginals
            kl_loss = logqz_prodmarginals - logpz
            
            if reduce == "mean":
                mi_loss = torch.mean(mi_loss)
                tc_loss = torch.mean(tc_loss)
                kl_loss = torch.mean(kl_loss)

            # recombine to get loss:
            return mi_loss + self.beta_kl * tc_loss + kl_loss
    
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

            loss_kl = self.compute_kl_loss(z, real_mu, real_logvar)
            loss = self.beta_rec * loss_rec + self.beta_kl * loss_kl

        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()

        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
            if self.clip:
                self.grad_scaler.unscale_(self.optimizer_e)
                self.grad_scaler.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip)
            self.grad_scaler.step(self.optimizer_e)
            self.grad_scaler.step(self.optimizer_d)
            self.grad_scaler.update()
        else:
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip)
            self.optimizer_e.step()
            self.optimizer_d.step()

        if torch.isnan(loss):
            raise RuntimeError

        if self.writer:
            self.write_scalars(
                cur_iter,
                losses=dict(
                    loss_rec=loss_rec.data.cpu().item(), loss_kl=loss_kl.data.cpu().item()
                ),
            )
            self._write_images_helper(real_batch, cur_iter)
            self.write_disentanglemnt_scores(cur_iter)