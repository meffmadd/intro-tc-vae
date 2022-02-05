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
from dataset import UkiyoE, load_labels, image_dir
import matplotlib.pyplot as plt
import matplotlib
from contextlib import nullcontext

from utils import *
from models import SoftIntroVAE

matplotlib.use("Agg")


def train_soft_intro_vae(
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
    num_vae=0,
    save_interval=50,
    optimizer="adam",
    recon_loss_type="mse",
    beta_kl=1.0,
    beta_rec=1.0,
    beta_neg=1.0,
    dropout=0.0,
    test_iter=1000,
    seed=-1,
    pretrained=None,
    device=torch.device("cpu"),
    num_row=8,
    gamma_r=1e-8,
    use_tensorboard=False,
    use_amp=False,
):
    """
    :param dataset: dataset to train on: ['ukiyo_e256', 'ukiyo_e128', 'ukiyo_e64', 'cifar10', 'fmnist']
    :param z_dim: latent dimensions
    :param lr_e: learning rate for encoder
    :param lr_d: learning rate for decoder
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param num_vae: number of epochs for vanilla vae training
    :param save_interval: epochs between checkpoint saving
    :param optimizer: the type of optimizer to use for training
    :param recon_loss_type: type of reconstruction loss ('mse', 'l1', 'bce')
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param dropout: fraction to use for dropout (0.0 means no dropout)
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param num_row: number of images in a row gor the sample image saving
    :param gamma_r: coefficient for the reconstruction loss for fake data in the decoder
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
    if dataset == "cifar10":
        image_size = 32
        channels = [64, 128, 256]
        train_set = CIFAR10(
            root="./cifar10_ds",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        ch = 3
    elif dataset == "fmnist":
        image_size = 28
        channels = [64, 128]
        train_set = FashionMNIST(
            root="./fmnist_ds",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        ch = 1
    elif dataset == "ukiyo_e256":
        image_size = 256
        channels = [64, 128, 256, 512, 512, 512]
        train_set = UkiyoE(image_dir, load_labels(), "Painter")
        ch = 3
    elif dataset == "ukiyo_e128":
        image_size = 128
        channels = [64, 128, 256, 512, 512]
        train_set = UkiyoE(image_dir, load_labels(), "Painter", resize=image_size)
        ch = 3
    elif dataset == "ukiyo_e64":
        image_size = 64
        channels = [64, 128, 256, 512]
        train_set = UkiyoE(image_dir, load_labels(), "Painter", resize=image_size)
        ch = 3
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

    scale = 1 / (
        ch * image_size ** 2
    )  # normalize by images size (channels * height * width)

    train_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    start_time = time.time()

    grad_scaler = torch.cuda.amp.GradScaler()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    exp_elbos_f = []
    exp_elbos_r = []
    for epoch in range(start_epoch, num_epochs):
        diff_kls = []
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = (
                dataset
                + "_soft_intro"
                + "_betas_"
                + str(beta_kl)
                + "_"
                + str(beta_neg)
                + "_"
                + str(beta_rec)
                + "_zdim_"
                + str(z_dim)
                + "_"
                + arch
                + "_"
                + optimizer
                + "_"
            )
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()

        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []
        batch_exp_elbo_f = []
        batch_exp_elbo_r = []

        pbar = tqdm(iterable=train_data_loader)

        for batch in pbar:
            # --------------train------------
            if dataset in [
                "cifar10",
                "fmnist",
                "ukiyo_e256",
                "ukiyo_e128",
                "ukiyo_e64",
            ]:
                batch = batch[0]
            if epoch < num_vae:
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                batch_size = batch.size(0)

                real_batch = batch.to(device)

                # =========== Update E, D ================
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    real_mu, real_logvar, z, rec = model(real_batch)

                    loss_rec = calc_reconstruction_loss(
                        real_batch, rec, loss_type=recon_loss_type, reduction="mean"
                    )
                    loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                    loss = beta_rec * loss_rec + beta_kl * loss_kl

                optimizer_d.zero_grad()
                optimizer_e.zero_grad()

                if use_amp:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer_e)
                    grad_scaler.step(optimizer_d)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer_e.step()
                    optimizer_d.step()

                pbar.set_description_str("epoch #{}".format(epoch))
                pbar.set_postfix(
                    r_loss=loss_rec.data.cpu().item(), kl=loss_kl.data.cpu().item()
                )

                if cur_iter % test_iter == 0:
                    vutils.save_image(
                        torch.cat([real_batch, rec], dim=0).data.cpu(),
                        "{}/image_{}.jpg".format(fig_dir, cur_iter),
                        nrow=num_row,
                    )

            else:
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                b_size = batch.size(0)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)

                real_batch = batch.to(device)

                # =========== Update E ================
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False

                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    fake = model.sample(noise_batch)

                    real_mu, real_logvar = model.encode(real_batch)
                    z = reparameterize(real_mu, real_logvar)
                    rec = model.decoder(z)

                    loss_rec = calc_reconstruction_loss(
                        real_batch, rec, loss_type=recon_loss_type, reduction="mean"
                    )

                    lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                    rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
                    fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

                    kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
                    kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

                    loss_rec_rec_e = calc_reconstruction_loss(
                        rec, rec_rec, loss_type=recon_loss_type, reduction="none"
                    )
                    while len(loss_rec_rec_e.shape) > 1:
                        loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                    loss_rec_fake_e = calc_reconstruction_loss(
                        fake, rec_fake, loss_type=recon_loss_type, reduction="none"
                    )
                    while len(loss_rec_fake_e.shape) > 1:
                        loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                    expelbo_rec = (
                        (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec))
                        .exp()
                        .mean()
                    )
                    expelbo_fake = (
                        (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake))
                        .exp()
                        .mean()
                    )

                    lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                    lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

                    lossE = lossE_real + lossE_fake

                optimizer_e.zero_grad()

                if use_amp:
                    grad_scaler.scale(lossE).backward()
                    grad_scaler.step(optimizer_e)
                else:
                    lossE.backward()
                    optimizer_e.step()

                # ========= Update D ==================
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    fake = model.sample(noise_batch)
                    rec = model.decoder(z.detach())
                    loss_rec = calc_reconstruction_loss(
                        real_batch, rec, loss_type=recon_loss_type, reduction="mean"
                    )

                    rec_mu, rec_logvar = model.encode(rec)
                    z_rec = reparameterize(rec_mu, rec_logvar)

                    fake_mu, fake_logvar = model.encode(fake)
                    z_fake = reparameterize(fake_mu, fake_logvar)

                    rec_rec = model.decode(z_rec.detach())
                    rec_fake = model.decode(z_fake.detach())

                    loss_rec_rec = calc_reconstruction_loss(
                        rec.detach(),
                        rec_rec,
                        loss_type=recon_loss_type,
                        reduction="mean",
                    )
                    loss_fake_rec = calc_reconstruction_loss(
                        fake.detach(),
                        rec_fake,
                        loss_type=recon_loss_type,
                        reduction="mean",
                    )

                    lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                    lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

                    lossD = scale * (
                        loss_rec * beta_rec
                        + (lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl
                        + gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_fake_rec)
                    )

                optimizer_d.zero_grad()

                if use_amp:
                    grad_scaler.scale(lossD).backward()
                    grad_scaler.step(optimizer_d)
                    grad_scaler.update()
                else:
                    lossD.backward()
                    optimizer_d.step()

                if torch.isnan(lossD) or torch.isnan(lossE):
                    raise SystemError

                dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
                pbar.set_description_str("epoch #{}".format(epoch))
                pbar.set_postfix(
                    r_loss=loss_rec.data.cpu().item(),
                    kl=lossE_real_kl.data.cpu().item(),
                    diff_kl=dif_kl.item(),
                    expelbo_f=expelbo_fake.cpu().item(),
                )

                if writer:
                    writer.add_scalars(
                        "losses",
                        dict(
                            r_loss=loss_rec.data.cpu().item(),
                            kl=lossE_real_kl.data.cpu().item(),
                            expelbo_f=expelbo_fake.cpu().item(),
                        ),
                        global_step=cur_iter,
                    )
                    writer.add_scalar("diff_kl", dif_kl.item(), global_step=cur_iter)
                    try:
                        writer.add_scalars(
                            "learning_rate",
                            dict(
                                e_lr=e_scheduler.get_last_lr()[0],
                                d_lr=d_scheduler.get_last_lr()[0],
                            ),
                            global_step=cur_iter,
                        )
                    except IndexError:
                        pass

                diff_kls.append(
                    -lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item()
                )
                batch_kls_real.append(lossE_real_kl.data.cpu().item())
                batch_kls_fake.append(lossD_fake_kl.cpu().item())
                batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
                batch_rec_errs.append(loss_rec.data.cpu().item())
                batch_exp_elbo_f.append(expelbo_fake.data.cpu())
                batch_exp_elbo_r.append(expelbo_rec.data.cpu())

                if cur_iter % test_iter == 0:
                    _, _, _, rec_det = model(real_batch, deterministic=True)
                    max_imgs = min(batch.size(0), 16)
                    vutils.save_image(
                        torch.cat(
                            [
                                real_batch[:max_imgs],
                                rec_det[:max_imgs],
                                fake[:max_imgs],
                            ],
                            dim=0,
                        ).data.cpu(),
                        "{}/image_{}.jpg".format(fig_dir, cur_iter),
                        nrow=num_row,
                    )
                    if writer:
                        writer.add_images(
                            f"image_{cur_iter}",
                            torch.cat(
                                [
                                    real_batch[:max_imgs],
                                    rec_det[:max_imgs],
                                    fake[:max_imgs],
                                ],
                                dim=0,
                            ).data.cpu(),
                        )

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

        if epoch > num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))
            exp_elbos_f.append(np.mean(batch_exp_elbo_f))
            exp_elbos_r.append(np.mean(batch_exp_elbo_r))
            # epoch summary
            print("#" * 50)
            print(f"Epoch {epoch} Summary:")
            print(f"beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}")
            print(
                f"rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}"
            )
            print(
                f"diff_kl: {np.mean(diff_kls):.3f}, exp_elbo_f: {exp_elbos_f[-1]:.4e}, exp_elbo_r: {exp_elbos_r[-1]:.4e}"
            )
            print(f"time: {time.time() - start_time}")
            print("#" * 50)
        if epoch == num_epochs - 1:
            with torch.no_grad():
                _, _, _, rec_det = model(real_batch, deterministic=True)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
                fake = model.sample(noise_batch)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(
                    torch.cat(
                        [real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]],
                        dim=0,
                    ).data.cpu(),
                    "{}/image_{}.jpg".format(fig_dir, cur_iter),
                    nrow=num_row,
                )
                if writer:
                    writer.add_images(
                        f"image_{cur_iter}",
                        torch.cat(
                            [
                                real_batch[:max_imgs],
                                rec_det[:max_imgs],
                                fake[:max_imgs],
                            ],
                            dim=0,
                        ).data.cpu(),
                    )

            # plot graphs
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
            ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
            ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
            ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
            ax.legend()

            plt.savefig(os.path.join(fig_dir, "soft_intro_train_graphs.jpg"))
            save_losses(fig_dir, kls_real, kls_fake, kls_rec, rec_errs)

            # save models
            prefix = (
                dataset
                + "_soft_intro"
                + "_betas_"
                + str(beta_kl)
                + "_"
                + str(beta_neg)
                + "_"
                + str(beta_rec)
                + "_zdim_"
                + str(z_dim)
                + "_"
                + arch
                + "_"
                + optimizer
                + "_"
            )
            save_checkpoint(model, epoch, cur_iter, prefix)
            model.train()

