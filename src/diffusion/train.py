import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import wandb

from diffusion.data.datasets import load_dataset, get_sample_shape
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
    # Create Training Dirs / Save Config
    # ----------
    train_dir = os.path.join(config.run.runs_dir, config.run.name, "training")
    os.makedirs(os.path.join(train_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'figs'), exist_ok=True)

    save_config(config, os.path.join(train_dir, 'config.yml'))

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
    # Set Device
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------
    # Initialize UNet Model
    # ----------
    sample_shape = get_sample_shape(config.data.dataset)

    model = UNet(
        in_ch=sample_shape[0], 
        base_ch=config.model.base_ch, 
        num_res_blocks=config.model.num_res_blocks, 
        ch_mults=config.model.ch_mults, 
        enc_heads=config.model.enc_heads, 
        mid_heads=config.model.mid_heads
    )
    model.to(device)

    # ----------
    # Create Diffusion Utilities Object
    # ----------
    diffusion = Diffusion(
        diffusion_config=config.diffusion, 
        sample_config=config.sample, 
        device=device
    )

    # ----------
    # Define Optimizer
    # ----------
    optimizer = torch.optim.Adam(model.parameters(), config.train.lr)

    # ----------
    # Initialize wandb Logging
    # ----------
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "diffusion"), 
        entity=os.environ.get("WANDB_ENTITY", None),
        name=config.run.name
    )

    # ----------
    # Create Trainer / Run Training
    # ----------
    trainer = Trainer(
        train_config=config.train, 
        sample_config=config.sample,
        model=model, 
        diffusion=diffusion, 
        optimizer=optimizer, 
        dataloader=dataloader, 
        device=device, 
        sample_shape=sample_shape, 
        train_dir=train_dir
    )
    trainer.train(config.train.epochs)

if __name__ == "__main__":
    main()