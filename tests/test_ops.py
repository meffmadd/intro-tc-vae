import sys
sys.path.append("..")

import pytest
import torch
import numpy as np
import ops


def test_calc_reconstruction_loss_mse():
    x = torch.from_numpy(np.array([0,0,0], dtype=np.float32))
    x_recon = torch.from_numpy(np.array([1,2,4], dtype=np.float32))
    loss = ops.reconstruction_loss(x, x_recon, loss_type="mse", reduction="sum")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0
    assert loss.item() == 21

    loss = ops.reconstruction_loss(x, x_recon, loss_type="mse", reduction="mean")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0
    assert loss.item() == 7

    loss = ops.reconstruction_loss(x, x_recon, loss_type="mse", reduction="none")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 1
    assert list(loss.numpy()) == [1,4,16]


def test_calc_reconstruction_loss_l1():
    x = torch.from_numpy(np.array([0,0,0], dtype=np.float32))
    x_recon = torch.from_numpy(np.array([1,2,4], dtype=np.float32))
    loss = ops.reconstruction_loss(x, x_recon, loss_type="l1", reduction="sum")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0
    assert loss.item() == 7

    loss = ops.reconstruction_loss(x, x_recon, loss_type="l1", reduction="mean")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0
    assert loss.item() == pytest.approx(7/3)

    loss = ops.reconstruction_loss(x, x_recon, loss_type="l1", reduction="none")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 2
    assert list(loss.flatten().numpy()) == [1,2,4]


def test_reparameterize():
    mu = torch.from_numpy(np.array([0,0,0], dtype=np.float32))
    logvar = torch.from_numpy(np.array([1,2,4], dtype=np.float32))
    sample = ops.reparameterize(mu, logvar)
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == mu.shape
    assert sample.shape == logvar.shape


def test_calc_kl():
    mu = torch.from_numpy(np.array([[0,0],[0,0]], dtype=np.float32))
    logvar = torch.from_numpy(np.array([[1,2],[4,8]], dtype=np.float32))
    loss = ops.kl_divergence(mu, logvar, reduce="sum")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0

    loss = ops.kl_divergence(mu, logvar, reduce="none")
    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 1
