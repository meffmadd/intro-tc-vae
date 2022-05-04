import sys
import os
sys.path.append("..")

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pickle
import train
from train import train_soft_intro_vae
import dataset
import utils


def test_train(monkeypatch, mocker):
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
    assert os.path.isdir(data_dir)
    monkeypatch.setattr(dataset, "data_dir", data_dir)

    image_dir = data_dir + "/arc_extracted_face_images"
    monkeypatch.setattr(dataset, "image_dir", image_dir)
    monkeypatch.setattr(train, "image_dir", image_dir) # imports not affected by monkey patching
    assert os.path.isdir(image_dir)
    assert len(dataset.UkiyoE(train.image_dir, dataset.load_labels(), "Painter", resize=64)) == 3
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "save_losses", lambda *args, **kwargs: None)
    monkeypatch.setattr(vutils, "save_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(pickle, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)

    optim_spy = mocker.spy(torch.optim.Adagrad, "__init__")

    train_soft_intro_vae(
        dataset="ukiyo_e64",
        z_dim=32,
        lr_e=2e-4,
        lr_d=2e-4,
        batch_size=3,
        num_workers=1,
        start_epoch=0,
        exit_on_negative_diff=False,
        num_epochs=1,
        num_vae=0,
        save_interval=50,
        recon_loss_type="mse",
        optimizer="adagrad",
        beta_kl=1.0,
        beta_rec=1.0,
        beta_neg=1.0,
        test_iter=1000,
        seed=-1,
        pretrained=None,
        device=torch.device("cpu"),
        num_row=8,
        gamma_r=1e-8,
        use_tensorboard=False,
    )

    assert optim_spy.call_count == 2


