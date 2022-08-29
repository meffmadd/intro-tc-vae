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
            clip,
        )

    def compute_kl_loss(
        self,
        z: Optional[Tensor],
        mu: Tensor,
        logvar: Tensor,
        reduce: str = "mean",
        beta: float = None,
    ) -> Tensor:
        if beta is None:
            beta = self.beta_kl

        dataset_size = len(self.dataset)

        kl_loss = kl_divergence(logvar, mu, reduce=reduce)
        tc = (beta - 1.0) * total_correlation(
            z, mu, logvar, dataset_size, reduce=reduce
        )
        return tc + kl_loss
