import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio
from torchvision.utils import make_grid


def make_sample_image(samples: torch.Tensor, save_path: str=None) -> np.ndarray:
    """
    Creates a grid image of a batch of generated samples.
    
    Args:
        samples (torch.Tensor): Tensor of generated samples of shape (B, C, H, W) in [-1,1]
        save_path (str): Save path for grid image
        
    Returns:
        image (np.ndarray): Image of sample grid
    """
    # ----------
    # Convert samples [-1, 1] -> [0, 1] / Create Grid
    # ----------
    samples = samples * 0.5 + 0.5
    samples = samples.clamp(0, 1)
    
    image = make_grid(samples, pad_value=1.0)
    
    # ----------
    # Convert grid [0, 1] -> [0, 255] / (C, H, W) -> (H, W, C)
    # ----------
    image = (image * 255).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).cpu().numpy()

    # ----------
    # Save grid image
    # ----------
    if save_path:
        iio.imwrite(save_path, image)
    
    return image


def make_sample_video(frames: list[torch.Tensor], fps: int, save_path: str=None) -> list[np.ndarray]:
    """
    Creates a video of a batch of generated sample trajectories.
    
    Args:
        frames (list[torch.Tensor]): List of frames of sample Tensors each of shape (B, C, H, W)
        save_path (str): Save path for mp4 video

    Returns:
        video_frames (list[np.ndarray]): List of video frame images
    """
    # ----------
    # Convert frames to grid images
    # ----------
    video_frames = [make_sample_image(frame) for frame in frames]

    # ----------
    # Upscale images to reduce compression
    # ----------
    video_frames = [upscale_image(img) for img in video_frames]

    # ----------
    # Save MP4
    # ----------
    if save_path:
        iio.imwrite(
            save_path, 
            video_frames, 
            fps=fps,
            codec="libx264"
        )

    return video_frames


def upscale_image(img: np.ndarray, scale: int=8) -> np.ndarray:
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img = F.interpolate(img, scale_factor=scale, mode="nearest")
    img = img.squeeze(0).permute(1, 2, 0).byte().numpy()
    return img