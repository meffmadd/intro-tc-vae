import torch
from torch import Tensor
from torchvision.utils import make_grid
import torch.nn.functional as F
import pickle

import os


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce="sum") -> torch.Tensor:
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = calc_kl_no_reduce(logvar, mu, mu_o, logvar_o)
    if reduce == "sum":
        kl = torch.sum(kl)
    elif reduce == "mean":
        kl = torch.mean(kl)
    return kl


@torch.jit.script
def calc_kl_no_reduce(
    logvar: Tensor, mu: Tensor, mu_o: Tensor, logvar_o: Tensor
) -> Tensor:
    kl = -0.5 * (
        1
        + logvar
        - logvar_o
        - logvar.exp() / torch.exp(logvar_o)
        - (mu - mu_o).pow(2) / torch.exp(logvar_o)
    ).sum(1)
    return kl


@torch.jit.script
def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std, device=device)
    return mu + eps * std


def calc_reconstruction_loss(
    x, recon_x, loss_type="mse", reduction="sum"
) -> torch.Tensor:
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ["sum", "mean", "none"]:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == "mse":
        recon_error = F.mse_loss(recon_x, x, reduction="none")
        recon_error = recon_error.sum(1)
        if reduction == "sum":
            recon_error = recon_error.sum()
        elif reduction == "mean":
            recon_error = recon_error.mean()
    elif loss_type == "l1":
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == "bce":
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights["model"], strict=False)


def save_losses(fig_dir, kls_real, kls_fake, kls_rec, rec_errs):
    with open(os.path.join(fig_dir, "soft_intro_train_graphs_data.pickle"), "wb") as fp:
        graph_dict = {
            "kl_real": kls_real,
            "kl_fake": kls_fake,
            "kl_rec": kls_rec,
            "rec_err": rec_errs,
        }
        pickle.dump(graph_dict, fp)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = (
        "./saves/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    )
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))
