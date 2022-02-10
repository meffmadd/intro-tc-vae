import torch
from torchvision.utils import make_grid
import pickle

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
