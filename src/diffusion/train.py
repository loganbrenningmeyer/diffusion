import os
import copy
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

def init_wandb(run_name: str):
    """
    Initializes wandb for logging, runs in offline mode on failure  
    """
    try:
        wandb.init(
            name=run_name,
            project=os.environ.get("WANDB_PROJECT", "diffusion"), 
            entity=os.environ.get("WANDB_ENTITY", None)
        )
    except Exception as e:
        # -- Use offline if init fails
        print(f"---- wandb.init() failed, running offline: {e}")
        wandb.init(
            name=run_name,
            mode='offline'
        )

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

    if not config.run.resume.enable:
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
    # Initialize UNet Model / EMA UNet Model
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

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.to(device)

    # ---------
    # Resume Training
    # ----------
    start_step = 1  # Default starting training step

    if config.run.resume.enable:
        ckpt_path = os.path.join(train_dir, "checkpoints", config.run.resume.ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])

        start_step = ckpt["step"] + 1  # Resume training step from saved checkpoint

        print(f"\n==== Resuming {config.run.resume.ckpt_name} training from step {ckpt['step']} ====")

    # ----------
    # Create Diffusion Utilities Object
    # ----------
    diffusion = Diffusion(
        diffusion_config=config.diffusion, 
        sample_config=config.sample, 
        sample_shape=sample_shape,
        device=device
    )

    # ----------
    # Define Optimizer
    # ----------
    optimizer = torch.optim.Adam(model.parameters(), config.train.lr)

    # ----------
    # Initialize wandb Logging
    # ----------
    if config.logging.wandb.enable:
        init_wandb(config.run.name)

    # ----------
    # Create Trainer / Run Training
    # ----------
    trainer = Trainer(
        model=model, 
        ema_model=ema_model,
        diffusion=diffusion, 
        optimizer=optimizer, 
        dataloader=dataloader, 
        device=device, 
        train_dir=train_dir,
        logging_config=config.logging,
        ema_decay=config.train.ema_decay,
        start_step=start_step
    )
    trainer.train(config.train.steps)

if __name__ == "__main__":
    main()