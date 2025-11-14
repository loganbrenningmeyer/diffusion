import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

from diffusion.data.datasets import load_dataset
from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion
from diffusion.training.trainer import Trainer


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def main():
    # ----------
    # Parse Arguments / Load Config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ----------
    # Set Device
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------
    # Initialize UNet Model
    # ----------
    in_ch = config.data.in_ch
    base_ch = config.model.base_ch
    ch_mults = config.model.ch_mults
    enc_heads = config.model.enc_heads
    mid_heads = config.model.mid_heads

    model = UNet(in_ch, base_ch, ch_mults, enc_heads, mid_heads)
    model.to(device)

    # ----------
    # Create Diffusion Utilities Object
    # ----------
    T = config.diffusion.T
    beta_1 = config.diffusion.beta_1
    beta_T = config.diffusion.beta_T
    beta_schedule = config.diffusion.beta_schedule
    sampler = config.diffusion.sampler
    n_steps = config.diffusion.n_steps
    eta = config.diffusion.eta

    diffusion = Diffusion(T, beta_1, beta_T, beta_schedule, sampler, n_steps, eta, device)

    # ----------
    # Define Optimizer
    # ----------
    lr = config.train.lr

    optimizer = torch.optim.Adam(model.parameters(), lr)

    # ----------
    # Create DataLoader
    # ----------
    dataset = load_dataset(config.data)

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # ----------
    # Initialize wandb Logging
    # ----------
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "diffusion"), 
        entity=os.environ.get("WANDB_ENTITY", None),
        name=config.train.run_name
    )

    # ----------
    # Create Trainer / Run Training
    # ----------
    os.makedirs('checkpoints/', exist_ok=True)

    trainer = Trainer(model, diffusion, optimizer, dataloader, device, config.train, config.data)
    trainer.train(config.train.epochs)

if __name__ == "__main__":
    main()