# imports
import torch
import argparse
from train import train_soft_intro_vae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        help="Loss function to use: ['vae','intro','tc','intro-tc']",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset to train on: ['ukiyo_e256', 'ukiyo_e128', 'ukiyo_e64', 'cifar10', 'fmnist']",
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=["conv", "res", "inception"],
        default="res",
        help="architecture choices for convolutional blocks in networks",
    )
    parser.add_argument(
        "-n",
        "--num_epochs",
        type=int,
        help="total number of epochs to run",
        default=250,
    )
    parser.add_argument(
        "-z", "--z_dim", type=int, help="latent dimensions", default=128
    )
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument(
        "-v",
        "--num_vae",
        type=int,
        help="number of epochs for vanilla vae training",
        default=0,
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        choices=["adam", "adadelta", "adagrad", "RMSprop"],
    )
    parser.add_argument(
        "-r",
        "--beta_rec",
        type=float,
        help="beta coefficient for the reconstruction loss",
        default=1.0,
    )
    parser.add_argument(
        "-k",
        "--beta_kl",
        type=float,
        help="beta coefficient for the kl divergence",
        default=1.0,
    )
    parser.add_argument(
        "-e",
        "--beta_neg",
        type=float,
        help="beta coefficient for the kl divergence in the expELBO function",
        default=1.0,
    )
    parser.add_argument(
        "-g",
        "--gamma_r",
        type=float,
        help="coefficient for the reconstruction loss for fake data in the decoder",
        default=1e-8,
    )
    parser.add_argument(
        "--dropout", type=float, help="dropout probability of an element", default=0.0
    )
    parser.add_argument("--seed", type=int, help="seed", default=-1)
    parser.add_argument(
        "-p",
        "--pretrained",
        type=str,
        help="path to pretrained model, to continue training",
        default="None",
    )
    parser.add_argument(
        "-c",
        "--device",
        type=int,
        help="device: -1 for cpu, 0 and up for specific cuda device",
        default=-1,
    )
    parser.add_argument("--tensorboard", action="store_true", help="enable tensorboard")
    parser.add_argument(
        "--amp", action="store_true", help="enable automatic multi precision"
    )
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if args.device <= -1
        else torch.device("cuda:" + str(args.device))
    )
    pretrained = None if args.pretrained == "None" else args.pretrained
    train_soft_intro_vae(
        solver_type=args.solver,
        dataset=args.dataset,
        arch=args.arch,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        num_workers=2,
        num_epochs=args.num_epochs,
        dropout=args.dropout,
        optimizer=args.optimizer,
        beta_kl=args.beta_kl,
        beta_neg=args.beta_neg,
        beta_rec=args.beta_rec,
        device=device,
        save_interval=100,
        start_epoch=0,
        lr_e=args.lr,
        lr_d=args.lr,
        pretrained=pretrained,
        seed=args.seed,
        test_iter=3000,
        use_tensorboard=args.tensorboard,
        use_amp=args.amp,
    )
