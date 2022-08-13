from typing import Union
import torch
from torchvision.utils import make_grid
import pickle
from torch.utils.tensorboard import SummaryWriter

import os


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


def check_non_finite_gradints(model):
    # check for non-finite gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            mask = torch.isfinite(param.grad)
            if not mask.all():
                print("Non-finite gradients in ", name, (~torch.isfinite(param.grad)).sum().cpu().item(), "values")


class LossDict(dict):
    def __add__(self, other: "LossDict") -> "LossDict":
        new = LossDict()
        keys = sorted(set(self.keys()) | set(other.keys()))
        for k in keys:
            new[k] = self.get(k, 0) + other.get(k, 0)
        return new
    
    def __truediv__(self, value: Union[int, float]) -> "LossDict":
        new = LossDict()
        for k, v in self.items():
            new[k] = v / value
        return new

class SingletonWriter(object):
    writer: SummaryWriter
    cur_iter: int
    test_iter: int

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SingletonWriter, cls).__new__(cls)
        return cls.instance
    
    @property
    def write_test_iter(self):
        return self.writer and self.cur_iter % self.test_iter == 0
