from dataclasses import dataclass

import json
import os
from typing import Optional

@dataclass
class Config:
    solver: str
    dataset: str
    arch: str
    optimizer: str
    recon_loss_type: str
    device: int

    lr: float
    batch_size: int
    num_epochs: int
    seed: int

    z_dim: int
    beta_rec: float
    beta_kl: float
    beta_neg: float
    gamma_r: float

    use_tensorboard: bool
    use_amp: bool
    profile: bool
    clip: Optional[float]
    anomaly_detection: bool

    num_workers: int
    save_interval: int
    start_epoch: int
    test_iter: int


_default_config = dict(
    solver=None,
    dataset=None,
    arch="res",
    optimizer="adam",
    recon_loss_type="mse",
    device=-1,
    lr=2e-4,
    batch_size=128,
    num_epochs=200,
    seed=-1,
    z_dim=32,
    beta_rec=1.0,
    beta_kl=1.0,
    beta_neg=1.0,
    gamma_r=1e-8,
    use_tensorboard=False,
    use_amp=True,
    profile=False,
    num_workers=2,
    save_interval=100,
    start_epoch=0,
    test_iter=5000,
    clip=None,
    anomaly_detection=False
)

def load_config(path: str, update_dict: dict) -> Config:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    with open(path, "r") as f:
        c = json.load(f)
    c = {**_default_config, **c, **update_dict} # update default config values with provided config
    return Config(**c)

