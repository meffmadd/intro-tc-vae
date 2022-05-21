# imports
# torch and friends
from functools import reduce
from config import Config
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
from dataset import DSprites, DSpritesSmall, UkiyoE, WrappedDataLoader
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
def train_soft_intro_vae(config: Config):
    """
    :param config: Config for a run
    :return:
    """
    if config.seed != -1:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", config.seed)

    device = (
        torch.device("cpu")
        if config.device <= -1
        else torch.device("cuda:" + str(config.device))
    )

    # run cudnn benchmark for optimal convolution algorithm
    torch.backends.cudnn.benchmark = True

    # --------------build models -------------------------
    if config.dataset == "ukiyo_e256":
        image_size = 256
        channels = [64, 128, 256, 512, 512, 512]
        train_set = UkiyoE.load_data()
        ch = 3
    elif config.dataset == "ukiyo_e128":
        image_size = 128
        channels = [64, 128, 256, 512, 512]
        train_set = UkiyoE.load_data(resize=image_size)
        ch = 3
    elif config.dataset == "ukiyo_e64":
        image_size = 64
        channels = [64, 128, 256, 512]
        train_set = UkiyoE.load_data(resize=image_size)
        ch = 3
    elif config.dataset == "dsprites":
        image_size = 64
        channels = [64, 128, 256, 512]
        train_set = DSprites.load_data()
        ch = 1
    elif config.dataset == "dsprites_small":
        image_size = 64
        channels = [64, 128, 256, 512]
        train_set = DSpritesSmall.load_data()
        ch = 1
    else:
        raise NotImplementedError("dataset is not supported")

    writer = (
        SummaryWriter(
            comment=f"_{config.solver}_{config.dataset}_z{config.z_dim}_{config.beta_kl}_{config.beta_neg}_{config.beta_rec}_{config.arch}_{config.optimizer}"
        )
        if config.use_tensorboard
        else None
    )

    model = SoftIntroVAE(
        arch=config.arch,
        cdim=ch,
        zdim=config.z_dim,
        channels=channels,
        image_size=image_size,
    ).to(device)
    print(model)
    print("{:,} Parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if config.anomaly_detection:
        torch.autograd.set_detect_anomaly(True) 
        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)

        def get_nan_hook(name):
            def nan_hook(self, _, output):
                if not isinstance(output, tuple):
                    outputs = [output]
                else:
                    outputs = output
                for i, out in enumerate(outputs):
                    nan_mask = torch.isnan(out)
                    if nan_mask.any():
                        print(f"In {name}: Found NAN in output {i}: {nan_mask.sum().item()}/{np.prod(np.array(output.shape))}")

            return nan_hook

        for name, submodule in model.named_modules():
            submodule.register_forward_hook(get_nan_hook(name))

    lr_e, lr_d = config.lr, config.lr
    if config.optimizer == "adam":
        optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e, eps=1e-4)
        optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d, eps=1e-4)
    elif config.optimizer == "adadelta":
        optimizer_e = optim.Adadelta(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.Adadelta(model.decoder.parameters(), lr=lr_d)
    elif config.optimizer == "adagrad":
        optimizer_e = optim.Adagrad(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.Adagrad(model.decoder.parameters(), lr=lr_d)
    elif config.optimizer == "RMSprop":
        optimizer_e = optim.RMSprop(model.encoder.parameters(), lr=lr_e)
        optimizer_d = optim.RMSprop(model.decoder.parameters(), lr=lr_d)
    else:
        raise ValueError("Unknown optimizer")

    _train_data_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    def batch_to_device(x: torch.Tensor, y: torch.Tensor):
        if config.anomaly_detection:
            assert x.max() <= 1.0
            assert x.min() >= 0.0
        return x.to(device), y.to(device)
    train_data_loader = WrappedDataLoader(_train_data_loader, batch_to_device)

    grad_scaler = torch.cuda.amp.GradScaler()

    solver_kwargs = dict(
        dataset=train_set,
        model=model,
        batch_size=config.batch_size,
        optimizer_e=optimizer_e,
        optimizer_d=optimizer_d,
        beta_kl=config.beta_kl,
        beta_rec=config.beta_rec,
        device=device,
        use_amp=config.use_amp,
        grad_scaler=grad_scaler,
        writer=writer,
        test_iter=config.test_iter,
        clip=config.clip,
    )
    if config.solver == "vae":
        solver = VAESolver(**solver_kwargs)
    elif config.solver == "intro":
        solver = IntroSolver(**solver_kwargs, beta_neg=config.beta_neg)
    elif config.solver == "tc":
        solver = TCSovler(**solver_kwargs)
    elif config.solver == "intro-tc":
        solver = IntroTCSovler(**solver_kwargs, beta_neg=config.beta_neg)
    else:
        raise ValueError(f"Solver '{config.solver_type}' not supported!")

    cur_iter = 0
    for epoch in range(config.start_epoch, config.num_epochs):
        # save models
        if epoch % config.save_interval == 0 and epoch > 0:
            save_epoch = (epoch // config.save_interval) * config.save_interval
            prefix = f"{config.solver}_{config.dataset}_betas_{str(config.beta_kl)}_{str(config.beta_neg)}_{str(config.beta_rec)}_zdim_{config.z_dim}_{config.arch}_{config.optimizer}"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()

        pbar = tqdm(iterable=train_data_loader)

        with torch.autograd.profiler.profile(enabled=config.profile) as prof:
            for batch in pbar:
                # --------------train------------
                if len(batch) == 2:  # (image,label) tuple
                    batch = batch[0]
                # Perform train step with specific loss funtion
                postfix = {}
                solver.train_step(batch, cur_iter)
                
                if config.anomaly_detection:
                    with torch.no_grad():
                        max = float("-inf")
                        for p in model.parameters():
                            norm = torch.sum(p.grad.data**2).item()
                            if norm > max:
                                max = norm
                    postfix.update({"L2": f"{max:.1f}"})

                pbar.set_postfix(postfix)
                if config.profile and cur_iter == 50:
                    break

                cur_iter += 1
        pbar.close()

        if config.profile:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            break

        if epoch == config.num_epochs - 1:
            b_size = batch.size(0)
            real_batch = batch.to(solver.device)
            noise_batch = torch.randn(size=(b_size, config.z_dim)).to(device)
            fake = model.sample(noise_batch)
            solver.write_images(real_batch, fake, cur_iter)

            # save models
            prefix = f"{config.solver}_{config.dataset}_betas_{str(config.beta_kl)}_{str(config.beta_neg)}_{str(config.beta_rec)}_zdim_{config.z_dim}_{config.arch}_{config.optimizer}"
            save_checkpoint(model, epoch, cur_iter, prefix)
            model.train()
