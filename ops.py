import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math

# https://github.com/carbonati/variational-zoo/blob/3a81967c3828fcdc5c0248e16e53533e931e00d7/vzoo/losses/ops.py#L30


def gaussian_log_density(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the log density of a Gaussian. This is just the simplified log of the PDF of a normal distribution."""
    log2pi = torch.log(Tensor([2.0 * math.pi])).to(x.device)
    inv_var = torch.exp(-logvar)
    delta = x - mu
    return -0.5 * (torch.square(delta) * inv_var + logvar + log2pi)


def batch_gaussian_density(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the gaussian log density between each sample in the batch.
    Takes in a 2D matrices of shape (batch_size, latent_dim) and returns a
    3D matrix of shape (batch_size, batch_size, latent_dim).
    Parameters

    Parameters
    ----------
    x : Tensor
        Reparameterized latent representation of shape (batch_size, latent_dim)
    mu : Tensor
        Mean latent tensor of shape (batch_size, latent_dim)
    logvar : Tensor
        Log variance latent tensor of shape (batch_size, latent_dim)

    Returns
    -------
    Tensor
        Gaussian log density matrix between each sample of shape (batch_size, batch_size, latent_dim)
    """
    batch_log_qz = gaussian_log_density(
        x=torch.unsqueeze(x, 1),
        mu=torch.unsqueeze(mu, 0),
        logvar=torch.unsqueeze(logvar, 0),
    )
    return batch_log_qz


def log_qz(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes log(q(z))"""
    log_prob_qz = batch_gaussian_density(x, mu, logvar)
    log_qz = torch.logsumexp(
        torch.sum(log_prob_qz, dim=2),
        dim=1,
    )
    return log_qz


def log_prod_qz_i(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    log_prob_qz = batch_gaussian_density(x, mu, logvar)
    log_prod_qzi = torch.sum(
        torch.logsumexp(log_prob_qz, dim=1),
        dim=1,
    )
    return log_prod_qzi


def log_qz_cond_x(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    log_qz_cond_x = torch.sum(
        gaussian_log_density(x, mu, logvar),
        dim=1,
    )
    return log_qz_cond_x


def log_pz(x: Tensor) -> Tensor:
    log_pz = torch.sum(
        gaussian_log_density(
            x,
            torch.zeros(x.shape, device=x.device),
            torch.zeros(x.shape, device=x.device),
        ),
        dim=1,
    )
    return log_pz


def on_off_diag(x: Tensor):
    """Computes the on and off diagonal of a tensor."""
    diag = torch.diagonal(x)
    off_diag = x - torch.diag_embed(x)
    return diag, off_diag


def entropy(x: np.ndarray, base=None, axis=0, eps=1e-9):
    """Calculates entropy for a sequence of classes or probabilities."""
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x has to be a numpy.ndarray object!")
    p = (x + eps) / np.sum(x + eps, axis=axis, keepdims=True)
    H = -np.sum(p * np.log(p + eps), axis=axis)
    if base is not None:
        H /= np.log(base + eps)
    return H


def kl_divergence(logvar: Tensor, mu: Tensor, reduce="sum") -> Tensor:
    """Calculate kl-divergence

    Parameters
    ----------
    logvar : Tensor
        log-variance from the encoder
    mu : Tensor
        mean from the encoder
    reduce : str, optional
        type of reduce: 'sum', 'none'

    Returns
    -------
    Tensor
        KL-Divergence
    """
    kl = kl_no_reduce(logvar, mu)
    if reduce == "sum":
        kl = torch.sum(kl)
    elif reduce == "mean":
        kl = torch.mean(kl)
    return kl


def kl_no_reduce(logvar: Tensor, mu: Tensor) -> Tensor:
    kl = -0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).sum(1)
    return kl


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)

    Parameters
    ----------
    mu : Tensor
        mean of x
    logvar : Tensor
        log variance of x

    Returns
    -------
    Tensor
        the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std, device=device)
    return mu + eps * std


def reconstruction_loss(x, recon_x, loss_type="mse", reduction="sum") -> Tensor:
    """Computes the reconstruction loss based on the loss type provided (default MSE)

    Parameters
    ----------
    x : _type_
        original inputs
    recon_x : _type_
        reconstruction of the VAE's input
    loss_type : str, optional
        "mse", "l1", "bce", by default "mse"
    reduction : str, optional
        "sum", "mean", "none", by default "sum"

    Returns
    -------
    Tensor
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    NotImplementedError
        _description_
    """
    batch_size = x.size(0)
    assert batch_size != 0

    if reduction not in ["sum", "mean", "none"]:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == "mse":
        recon_error = F.mse_loss(recon_x, x, reduction=reduction).div(batch_size)
    elif loss_type == "l1":
        recon_error = F.l1_loss(recon_x, x, reduction=reduction).div(batch_size)
    elif loss_type == "bce":
        recon_error = F.binary_cross_entropy_with_logits(recon_x, x, reduction=reduction).div(batch_size)
    else:
        raise NotImplementedError
    return recon_error
