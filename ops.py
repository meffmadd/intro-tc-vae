import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math

# https://github.com/carbonati/variational-zoo/blob/3a81967c3828fcdc5c0248e16e53533e931e00d7/vzoo/losses/ops.py#L30
# https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/utils/math.py#L34
# https://github.com/julian-carpenter/beta-TCVAE/blob/572d9e31993ccce47ef7a072a49c027c9c944e5e/nn/losses.py#L79
# https://github.com/clementchadebec/benchmark_VAE/blob/d8a3c21594f77182655d241a8632bbe772f67e3e/src/pythae/models/beta_tc_vae/beta_tc_vae_model.py#L177
# https://github.com/rtqichen/beta-tcvae/blob/1a3577dbb14642b9ac27010928d12132d0c0fb91/lib/dist.py#L50
# https://github.com/nmichlo/disent/blob/67ed5b92aeef247f1c0cb3b8597e9fd95e69e817/disent/frameworks/vae/_unsupervised__betatcvae.py#L115


def gaussian_log_density_torch(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the log density of a Gaussian. This is just the simplified log of the PDF of a normal distribution."""
    var = torch.exp(logvar)
    log_prob = -F.gaussian_nll_loss(x, mu, var, reduction="none", eps=1e-3, full=True)
    # only clamp return value (don't use large eps) since we may still have small (x-mu)/var
    # semantically log probabilities of -50 or -5000 make no difference (still 0)
    return torch.clamp(log_prob, min=-50)


def batch_gaussian_density_torch(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
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
    batch_log_qz = gaussian_log_density_torch(
        x=torch.unsqueeze(x, 1),
        mu=torch.unsqueeze(mu, 0),
        logvar=torch.unsqueeze(logvar, 1),
    )
    return batch_log_qz


# references:
# https://github.com/carbonati/variational-zoo/blob/3a81967c3828fcdc5c0248e16e53533e931e00d7/vzoo/losses/ops.py#L59
# https://github.com/clementchadebec/benchmark_VAE/blob/d8a3c21594f77182655d241a8632bbe772f67e3e/src/pythae/models/beta_tc_vae/beta_tc_vae_model.py#L142
# https://github.com/rtqichen/beta-tcvae/blob/1a3577dbb14642b9ac27010928d12132d0c0fb91/vae_quant.py#L227
# https://github.com/nmichlo/disent/blob/67ed5b92aeef247f1c0cb3b8597e9fd95e69e817/disent/frameworks/vae/_unsupervised__betatcvae.py#L93
# https://github.com/julian-carpenter/beta-TCVAE/blob/572d9e31993ccce47ef7a072a49c027c9c944e5e/nn/losses.py#L93

def log_qz(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes log(q(z))"""
    log_prob_qz = batch_gaussian_density_torch(x, mu, logvar)  # log prob between (-inf, 0]
    # interesting: https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/evaluate.py#L288
    log_qz = torch.logsumexp(
        torch.sum(log_prob_qz, dim=2),
        dim=1,
    )
    return log_qz


def log_prod_qz_i(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    log_prob_qz = batch_gaussian_density_torch(x, mu, logvar)
    log_prod_qzi = torch.sum(
        torch.logsumexp(log_prob_qz, dim=1),
        dim=1,
    )
    return log_prod_qzi


def log_qz_cond_x(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    log_qz_cond_x = torch.sum(
        gaussian_log_density_torch(x, mu, logvar),
        dim=1,
    )
    return log_qz_cond_x


def log_pz(x: Tensor) -> Tensor:
    log_pz = torch.sum(
        gaussian_log_density_torch(
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
    x = x.view(x.size(0), -1).detach()
    if loss_type == "mse":
        recon_error = F.mse_loss(recon_x, x, reduction=reduction)
    elif loss_type == "l1":
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == "bce":
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error
