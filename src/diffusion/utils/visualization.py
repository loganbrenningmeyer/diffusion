import os
import imageio
import numpy as np
import torch
from torchvision.utils import make_grid


def make_sample_grid(x: torch.Tensor):
    """
    Creates a grid of generated samples for visualization.
    
    Args:
        x (Tensor): Tensor of generated samples of shape (B, C, H, W) in [-1,1]
    
    Returns:
        grid (Tensor): Tensor image of sample grid
    """
    # ----------
    # Convert x [-1,1] -> [0,1] / Create Grid
    # ----------
    x = x * 0.5 + 0.5
    x = x.clamp(0.0, 1.0)
    
    grid = make_grid(x, pad_value=1.0).cpu()
    
    return grid

def make_sample_video(X: torch.Tensor, save_dir: str):
    """
    
    
    Args:
        X (Tensor): Tensor of samples over timesteps of shape (B, T, C, H, W)
        save_dir (str): Save directory for mp4 video

    Returns:
        save_paths (list[str]): List of video save paths
    """
    os.makedirs(save_dir, exist_ok=True)

    save_paths = []

    for i, x in enumerate(X):
        # ----------
        # Create imageio writer
        # ----------
        save_path = os.path.join(save_dir, f'sample_{i}.mp4')
        writer = imageio.get_writer(save_path, fps=24, codec='libx264', format='FFMPEG')
        save_paths.append(save_path)

        # ----------
        # Permute dimensions
        # ----------
        x = x.detach().cpu()
        x = x.permute(0, 2, 3, 1)   # (T, H, W, C)

        for x_t in x:
            # ----------
            # Convert x_t [-1,1] -> [0,255]
            # ----------
            x_t = (x_t * 0.5 + 0.5) * 255
            x_t = x_t.clamp(0, 255).numpy().astype(np.uint8)

            writer.append_data(x_t)

        writer.close()

    return save_paths