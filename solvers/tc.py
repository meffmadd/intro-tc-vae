from dataset import DisentanglementDataset
from solvers.vae import VAESolver
from typing import Optional
from models import SoftIntroVAE
from ops import (
    gaussian_log_density,
    minibatch_stratified_sampling,
    minibatch_weighted_sampling,
)

from contextlib import nullcontext
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils import SingletonWriter


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

        batch_size = z.size(0)
        dataset_size = len(self.dataset)

        # calculate log q(z|x)
        logqz_condx = gaussian_log_density(z, mu, logvar).sum(dim=1)

        # calculate log p(z)
        # mean and log var is 0
        zeros = torch.zeros_like(z)
        logpz = gaussian_log_density(z, zeros, zeros).sum(dim=1)

        log_qz_prob = gaussian_log_density(
            z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0)
        )
        logqz_prodmarginals, log_qz = minibatch_stratified_sampling(
            log_qz_prob, batch_size, dataset_size
        )

        mi_loss = logqz_condx - log_qz
        tc_loss = log_qz - logqz_prodmarginals
        kl_loss = logqz_prodmarginals - logpz

        if reduce == "mean":
            mi_loss = torch.mean(mi_loss)
            tc_loss = torch.mean(tc_loss)
            kl_loss = torch.mean(kl_loss)

            if SingletonWriter().writer and reduce == "mean":
                SingletonWriter().writer.add_scalars(
                    "tc_decomp",
                    {   
                        "mi": mi_loss.data.item(),
                        "tc": tc_loss.data.item(),
                        "kl": kl_loss.data.item()
                    },
                    global_step=SingletonWriter().cur_iter,
                )

        # recombine to get loss:
        return mi_loss + beta * tc_loss + kl_loss
