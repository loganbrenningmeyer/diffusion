import argparse
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

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
    in_ch     = config.data.in_ch
    base_ch   = config.model.base_ch
    ch_mults  = config.model.ch_mults
    enc_heads = config.model.enc_heads
    mid_heads = config.model.mid_heads

    model = UNet(in_ch, base_ch, ch_mults, enc_heads, mid_heads)
    model.to(device)

    # ----------
    # Create Diffusion Utilities Object
    # ----------
    timesteps     = config.diffusion.timesteps
    beta_start    = config.diffusion.beta_start
    beta_end      = config.diffusion.beta_end
    beta_schedule = config.diffusion.beta_schedule

    diffusion = Diffusion(timesteps, beta_start, beta_end, beta_schedule, device)

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
    # Create Trainer / Run Training
    # ----------
    trainer = Trainer(model, diffusion, dataloader, optimizer, device, config.train)

    epochs = config.train.epochs
    trainer.train(epochs)

if __name__ == "__main__":
    main()