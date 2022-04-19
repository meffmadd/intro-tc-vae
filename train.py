# imports
# torch and friends
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# standard
import os
import random
import time
import numpy as np
from tqdm import tqdm
from dataset import DSprites, UkiyoE
import matplotlib.pyplot as plt
import matplotlib
from contextlib import nullcontext
from solvers import VAESolver, IntroSolver
from solvers.intro_tc import IntroTCSovler
from solvers.tc import TCSovler

from utils import *
from models import SoftIntroVAE

matplotlib.use("Agg")

# TODO: numpy docstring type
def train_soft_intro_vae(
    solver_type="vae",
    dataset="cifar10",
    arch="res",
    z_dim=128,
    lr_e=2e-4,
    lr_d=2e-4,
    batch_size=128,
    num_workers=4,
    start_epoch=0,
    exit_on_negative_diff=False,
    num_epochs=250,
    save_interval=50,
    optimizer="adam",
    beta_kl=1.0,
    beta_rec=1.0,
    beta_neg=1.0,
    dropout=0.0,
    test_iter=1000,
    seed=-1,
    pretrained=None,
    device=torch.device("cpu"),
    use_tensorboard=False,
    use_amp=False,
):
    """
    :param solver_type: the type of objective function to be optimized: ['vae','intro','tc','intro-tc']
    :param dataset: dataset to train on: ['ukiyo_e256', 'ukiyo_e128', 'ukiyo_e64', 'cifar10', 'fmnist']
    :param arch: model architecture: ['conv', 'res', 'inception']
    :param z_dim: number of latent dimensions
    :param lr_e: learning rate for encoder
    :param lr_d: learning rate for decoder
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param save_interval: epochs between checkpoint saving
    :param optimizer: the type of optimizer to use for training
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param dropout: fraction to use for dropout (0.0 means no dropout)
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param use_tensorboard: whether to write the model loss and output to be used for tensorboard
    :param use_amp: whether to use automatic multi precision
    :return:
    """
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # run cudnn benchmark for optimal convolution computation
    torch.backends.cudnn.benchmark = True

    # --------------build models -------------------------
    if dataset == "ukiyo_e256":
        image_size = 256
        channels = [64, 128, 256, 512, 512, 512]
        train_set = UkiyoE.load_data()
        ch = 3
    elif dataset == "ukiyo_e128":
        image_size = 128
        channels = [64, 128, 256, 512, 512]
        train_set = UkiyoE.load_data(resize=image_size)
        ch = 3
    elif dataset == "ukiyo_e64":
        image_size = 64
        channels = [64, 128, 256, 512]
        train_set = UkiyoE.load_data(resize=image_size)
        ch = 3
    elif dataset == "dsprites":
        image_size = 64
        channels = [64, 128, 256, 512]
        train_set = DSprites.load_data()
        ch = 1
    else:
        raise NotImplementedError("dataset is not supported")

    writer = (
        SummaryWriter(
            comment=f"_{dataset}_z{z_dim}_{beta_kl}_{beta_neg}_{beta_rec}_{arch}_{optimizer}_"
        )
        if use_tensorboard
        else None
    )

    model = SoftIntroVAE(
        arch=arch,
        cdim=ch,
        zdim=z_dim,
        channels=channels,
        image_size=image_size,
        dropout=dropout,
    ).to(device)
    if pretrained is not None:
        load_model(model, pretrained, device)
    print(model)

    fig_dir = "./figures_" + dataset
    os.makedirs(fig_dir, exist_ok=True)

    if optimizer == "adam":
        optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)
    elif optimizer == "adadelta":
        optimizer_e = optim.Adadelta(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.Adadelta(model.decoder.parameters(), lr=lr_d)
    elif optimizer == "adagrad":
        optimizer_e = optim.Adagrad(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.Adagrad(model.decoder.parameters(), lr=lr_d)
    elif optimizer == "RMSprop":
        optimizer_e = optim.RMSprop(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.RMSprop(model.decoder.parameters(), lr=lr_d)
    else:
        raise ValueError("Unknown optimizer")

    e_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer_e, milestones=(350,), gamma=0.1
    )
    d_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer_d, milestones=(350,), gamma=0.1
    )

    train_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    grad_scaler = torch.cuda.amp.GradScaler()

    solver_kwargs = dict(
        dataset=train_set,
        model=model,
        batch_size=batch_size,
        optimizer_e=optimizer_e,
        optimizer_d=optimizer_d,
        beta_kl=beta_kl,
        beta_rec=beta_rec,
        device=device,
        use_amp=use_amp,
        grad_scaler=grad_scaler,
        writer=writer,
        test_iter=test_iter,
    )
    if solver_type == "vae":
        solver = VAESolver(**solver_kwargs)
    elif solver_type == "intro":
        solver = IntroSolver(**solver_kwargs, beta_neg=beta_neg)
    elif solver_type == "tc":
        solver = TCSovler(**solver_kwargs)
    elif solver_type == "intro-tc":
        solver = IntroTCSovler(**solver_kwargs, beta_neg=beta_neg)
    else:
        raise ValueError(f"Solver '{solver_type}' not supported!")

    cur_iter = 0
    for epoch in range(start_epoch, num_epochs):
        diff_kls = []
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = f"{dataset}_{solver}_betas_{str(beta_kl)}_{str(beta_neg)}_{str(beta_rec)}_zdim_{z_dim}_{arch}_{optimizer}"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()

        pbar = tqdm(iterable=train_data_loader)

        for batch in pbar:
            # --------------train------------
            if dataset in [
                "ukiyo_e256",
                "ukiyo_e128",
                "ukiyo_e64",
                "dsprites"
            ]:
                batch = batch[0]
            # Perform train step with specific loss funtion
            solver.train_step(batch, cur_iter)

            cur_iter += 1
        e_scheduler.step()
        d_scheduler.step()
        pbar.close()
        if exit_on_negative_diff and epoch > 50 and np.mean(diff_kls) < -1.0:
            print(
                f"the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)"
            )
            print("try to lower beta_neg hyperparameter")
            print("exiting...")
            raise SystemError("Negative KL Difference")

        if epoch == num_epochs - 1:
            b_size = batch.size(0)
            real_batch = batch.to(solver.device)
            noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
            fake = model.sample(noise_batch)
            solver.write_images(real_batch, fake, cur_iter)

            # save models
            prefix = f"{dataset}_{solver}_betas_{str(beta_kl)}_{str(beta_neg)}_{str(beta_rec)}_zdim_{z_dim}_{arch}_{optimizer}"
            save_checkpoint(model, epoch, cur_iter, prefix)
            model.train()
