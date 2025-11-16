import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import wandb

from diffusion.data.datasets import load_dataset
from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion
from diffusion.training.trainer import Trainer


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)

def main():
    # ----------
    # Parse Arguments / Load Config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ---------
    # Create Run Dirs / Save Config
    # ----------
    run_dir = os.path.join(config.train.logging.save_dir, config.train.run_name)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'figs'), exist_ok=True)

    save_config(config, os.path.join(run_dir, 'config.yml'))

    # ----------
    # Create DataLoader
    # ----------
    dataset, sample_shape = load_dataset(config.data)

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # ----------
    # Set Device
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------
    # Initialize UNet Model
    # ----------
    in_ch = sample_shape[0]
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
    diffusion = Diffusion(config.diffusion, device)

    # ----------
    # Define Optimizer / Scheduler
    # ----------
    optimizer = torch.optim.Adam(model.parameters(), config.train.lr)

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
    trainer = Trainer(config.train, model, diffusion, optimizer, dataloader, device, sample_shape)
    trainer.train(config.train.epochs)

if __name__ == "__main__":
    main()