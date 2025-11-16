import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from diffusion.data.datasets import load_dataset
from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion
from diffusion.utils.visualization import make_sample_grid


def load_config(config_path: str) -> DictConfig:
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
    num_res_blocks = config.model.num_res_blocks
    ch_mults = config.model.ch_mults
    enc_heads = config.model.enc_heads
    mid_heads = config.model.mid_heads

    model = UNet(in_ch, base_ch, num_res_blocks, ch_mults, enc_heads, mid_heads)
    model.to(device)

    # ----------
    # Create Diffusion Utilities Object
    # ----------
    T = config.diffusion.T
    beta_1 = config.diffusion.beta_1
    beta_T = config.diffusion.beta_T
    beta_schedule = config.diffusion.beta_schedule
    sampler = config.diffusion.sampler
    ddim_steps = config.diffusion.ddim_steps
    eta = config.diffusion.eta

    diffusion = Diffusion(T, beta_1, beta_T, beta_schedule, sampler, ddim_steps, eta, device)

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
    # Visualize Batch
    # ----------

if __name__ == "__main__":
    main()