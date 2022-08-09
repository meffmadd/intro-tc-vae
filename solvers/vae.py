from contextlib import nullcontext
from typing import Optional, Tuple

from matplotlib.lines import Line2D
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
import matplotlib.pyplot as plt
import numpy as np

from ops import kl_divergence, reconstruction_loss


class VAESolver:
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
        self.clip = clip
        self.recon_loss_type = recon_loss_type
    
    def compute_kl_loss(self, z: Optional[Tensor], mu: Tensor, logvar: Tensor, reduce: str = "mean") -> Tensor:
        return kl_divergence(logvar, mu, reduce=reduce)

    def train_step(self, batch: Tensor, cur_iter: int) -> float:
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real_batch = batch.to(self.device)

        # =========== Update E, D ================
        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            real_mu, real_logvar, z, rec = self.model(real_batch)

            loss_rec = reconstruction_loss(
                real_batch, rec, loss_type=self.recon_loss_type, reduction="mean"
            )
            loss_kl = self.compute_kl_loss(None, real_mu, real_logvar)

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
                    loss_rec=self.beta_rec * loss_rec.data.cpu().item(), loss_kl= self.beta_kl * loss_kl.data.cpu().item()
                ),
            )
            self.write_gradient_norm(cur_iter)
            self._write_images_helper(real_batch, cur_iter)
            self.write_disentanglemnt_scores(cur_iter)
            self.writer.flush()
        
        return loss

    def _write_images_helper(self, batch, cur_iter):
        if self.writer is not None and cur_iter % self.test_iter == 0:
            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, self.model.zdim), device=self.device)
            fake = self.model.sample(noise_batch).to(self.device)
            self.write_images(batch, fake, cur_iter)

    def write_images(self, batch, fake_batch, cur_iter):
        if self.writer is not None and cur_iter % self.test_iter == 0:
            with torch.no_grad():
                _, _, _, rec_det = self.model(batch, deterministic=True)
                max_imgs = min(batch.size(0), 16)
                self.writer.add_images(
                    "reconstructions",
                    torch.cat(
                        [
                            batch[:max_imgs],
                            rec_det[:max_imgs],
                            fake_batch[:max_imgs],
                        ],
                        dim=0,
                    ).data.cpu(),
                    global_step=cur_iter
                )

    def write_gradient_norm(self, cur_iter: int):
        # if self.writer:
        parameters = self.model.encoder.fc.parameters()
        total_norm: Tensor = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        self.writer.add_scalar("fc_grad_norm", total_norm.item(), global_step=cur_iter)


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
            if training := self.model.training:
                self.model.eval()
                
            if len(self.dataset) < num_samples:
                num_samples = len(self.dataset) // 2
            score_kwargs = dict(
                latent_generator=self.latent_generator,
                model=self.model,
                num_samples=num_samples,
                batch_size=self.batch_size,
            )
            print("Calculating disentanglment scores...")
            write_bvae_score(self.writer, cur_iter, **score_kwargs)
            write_dci_score(self.writer, cur_iter, **score_kwargs)
            write_mig_score(self.writer, cur_iter, **score_kwargs)
            write_mod_expl_score(self.writer, cur_iter, **score_kwargs)

            if training:
                self.model.train()
            print("Finished calculating disentanglemnt scores!")
    
    def write_gradient_flow(self, cur_iter, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        From: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "write_gradient_flow(cur_iter, self.model.named_parameters())" to visualize the gradient flow'''
        if self.writer is None or cur_iter % self.test_iter != 0:
            return

        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.cpu().abs().mean())
                max_grads.append(p.grad.cpu().abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        assert len(plt.get_fignums()) != 0
        fig = plt.gcf()
        self.writer.add_figure("gradient_flow", fig, global_step=cur_iter, close=True)
