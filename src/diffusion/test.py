import os
import argparse
import torch
from omegaconf import OmegaConf, DictConfig

from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion
from diffusion.data.datasets import get_sample_shape
from diffusion.utils.visualization import make_sample_image, make_sample_video


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

    test_config = load_config(args.config)

    train_dir = os.path.join(test_config.run.run_dir, "training")
    train_config = load_config(os.path.join(train_dir, "config.yml"))

    # ---------
    # Create Testing Dirs / Save Config
    # ----------
    test_dir = os.path.join(test_config.run.run_dir, "testing", test_config.run.name)
    os.makedirs(os.path.join(test_dir, 'figs'), exist_ok=True)

    save_config(test_config, os.path.join(test_dir, 'config.yml'))

    # ----------
    # Set Device
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------
    # Load UNet Model
    # ----------
    sample_shape = get_sample_shape(train_config.data.dataset)
    
    model = UNet(
        in_ch=sample_shape[0],
        base_ch=train_config.model.base_ch,
        num_res_blocks=train_config.model.num_res_blocks,
        ch_mults=train_config.model.ch_mults,
        enc_heads=train_config.model.enc_heads,
        mid_heads=train_config.model.mid_heads
    )

    ckpt_path = os.path.join(train_dir, "checkpoints", test_config.run.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model.load_state_dict(ckpt["ema_model"])
    model.to(device)
    model.eval()

    # ----------
    # Create Diffusion Utilities Object
    # ----------
    diffusion = Diffusion(
        diffusion_config=train_config.diffusion, 
        sample_config=test_config.sample, 
        device=device
    )

    # ---------
    # Generate / Save Samples
    # ----------
    samples_shape = (test_config.sample.num_samples, *sample_shape)

    frames = diffusion.sample_frames(model, samples_shape, test_config.sample.num_frames)
    samples = frames[-1]

    video_path = os.path.join(test_dir, "figs", "trajectory.mp4")
    image_path = os.path.join(test_dir, "figs", "samples.png")

    make_sample_video(frames, test_config.sample.fps, video_path)
    make_sample_image(samples, image_path)

if __name__ == "__main__":
    main()