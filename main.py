import argparse
import json
from config import load_config
from train import train_soft_intro_vae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument(
        "-f",
        "--config",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "-u",
        "--update",
        type=json.loads,
        default="{}",
        help="Path to the config file",
    )
    args = parser.parse_args()
    config = load_config(args.config, update_dict=args.update)
    train_soft_intro_vae(config=config)
