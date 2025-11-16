import torch
import numpy as np
import imageio.v3 as iio
from torchvision.utils import make_grid


def make_sample_grid(samples: torch.Tensor, save_path: str=None) -> np.ndarray:
    """
    Creates a grid image of a batch of generated samples.
    
    Args:
        samples (torch.Tensor): Tensor of generated samples of shape (B, C, H, W) in [-1,1]
    
    Returns:
        grid (np.ndarray): Image of sample grid
    """
    # ----------
    # Convert samples [-1, 1] -> [0, 1] / Create Grid
    # ----------
    samples = samples * 0.5 + 0.5
    samples = samples.clamp(0, 1)
    
    grid = make_grid(samples, pad_value=1.0)
    
    # ----------
    # Convert grid [0, 1] -> [0, 255] / (C, H, W) -> (H, W, C)
    # ----------
    grid = (grid * 255).clamp(0, 255).to(torch.uint8)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    # ----------
    # Save grid image
    # ----------
    if save_path:
        iio.imwrite(save_path, grid)
    
    return grid

def make_sample_video(frames: list[torch.Tensor], save_path: str=None) -> list[np.ndarray]:
    """
    Creates a video of a batch of generated sample trajectories.
    
    Args:
        frames (list[torch.Tensor]): List of frames of sample Tensors each of shape (B, C, H, W)
        save_path (str): Save path for mp4 video

    Returns:
        grid_frames (list[np.ndarray]): List of video frames
    """
    # ----------
    # Convert frames to grid images
    # ----------
    grid_frames = [make_sample_grid(frame) for frame in frames]

    # ----------
    # Save MP4
    # ----------
    if save_path:
        iio.imwrite(save_path, grid_frames, fps=10, codec="libx264")

    return grid_frames