import torch
from torchvision.utils import make_grid, save_image


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
    x = x.clamp(0,1)
    
    grid = make_grid(x, pad_value=1.0).cpu()
    return grid