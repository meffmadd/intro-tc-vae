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


def gaussian_log_density(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    normalization = torch.log(Tensor([2.0 * math.pi])).to(x.device)
    inv_sigma = torch.exp(-logvar)
    tmp = x - mu
    log_prob = -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)
    return torch.clamp(log_prob, min=-50)


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix
    Parameters
    ----------
    batch_size: int
        number of samples in a batch
    dataset_size: int
        number of samples in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()


def total_correlation(
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    dataset_size: int,
    reduce: str = "mean",
) -> Tensor:
    """Estimate of total correlation on a batch.
    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)
    Args:
        z: [batch_size, num_latents]-tensor with sampled representation.
        mu: [batch_size, num_latents]-tensor with mean of the encoder.
        logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
        dataset_size: number of samples in the dataset
        reduce: reduction type (either "none" or "mean")
    Returns:
        Total correlation estimated on a batch.

    Reference implementation is from: https://github.com/google-research/disentanglement_lib
    """
    batch_size = z.size(0)

    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = gaussian_log_density(
        z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0)
    )

    log_qz_product, log_qz = minibatch_stratified_sampling(log_qz_prob, batch_size, dataset_size)

    if reduce == "mean":
        return torch.mean(log_qz - log_qz_product)
    else:
        return log_qz - log_qz_product


def minibatch_weighted_sampling(log_qz_prob: Tensor, batch_size: int, dataset_size: int):
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    logqz_prodmarginals = (torch.logsumexp(log_qz_prob, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(dim=1)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = torch.logsumexp(log_qz_prob.sum(dim=2), dim=1, keepdim=False) - math.log(batch_size * dataset_size)

    return logqz_prodmarginals, log_qz


def minibatch_stratified_sampling(log_qz_prob: Tensor, batch_size: int, dataset_size: int):
    log_iw_mat = log_importance_weight_matrix(batch_size, dataset_size).to(log_qz_prob.device)

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    logqz_prodmarginals = torch.logsumexp(log_iw_mat.view(batch_size, batch_size, 1) + log_qz_prob, dim=1, keepdim=False).sum(dim=1)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = torch.logsumexp(log_iw_mat + log_qz_prob.sum(dim=2), dim=1, keepdim=False)

    return logqz_prodmarginals, log_qz


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
