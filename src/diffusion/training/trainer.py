import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion
from diffusion.utils.visualization import make_sample_image, make_sample_video


class Trainer:
    """
    Conducts the training process for a diffusion UNet model.

    Provides a `train()` loop and `train_step()` method for 
    forward noising and optimizing the MSE denoising objective.
    
    Args:
        model (UNet): The UNet model used for noise prediction / training.
        diffusion (Diffusion): The Diffusion helper object for handling diffusion functionality.
            - noise scheduling, forward noising, sampling
        optimizer (Optimizer): The optimizer used for training the model.
        dataloader (DataLoader): Training dataset DataLoader providing batches of images.
        device (torch.device): Device on which training will be performed.
        train_config (DictConfig): Config object containing misc. training parameters.
            - EMA decay, logging interval, saving interval
        data_config (DictConfig): Config object containing dataset information for sampling
            - in_ch, image_size
    """
    def __init__(
            self, 
            model: UNet, 
            diffusion: Diffusion, 
            optimizer: Optimizer, 
            dataloader: DataLoader, 
            device: torch.device, 
            train_config: DictConfig,
            data_config: DictConfig
    ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

        # -- Logging Parameters
        self.log_interval = train_config.logging.log_interval
        self.ckpt_interval = train_config.logging.ckpt_interval
        self.sample_interval = train_config.logging.sample_interval
        self.save_dir = train_config.logging.save_dir
        self.run_name = train_config.run_name

        # -- Sampling Parameters
        num_samples = train_config.sampling.num_samples
        in_ch = data_config.in_ch
        image_size = data_config.image_size
        self.sample_shape = (num_samples, in_ch, image_size, image_size)
        self.num_frames = train_config.sampling.num_frames
        self.fps = train_config.sampling.fps

        # ----------
        # Initialize EMA Model
        # ----------
        self.ema_decay = train_config.ema_decay

        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    def train(self, epochs: int):
        """
        Trains the UNet model for the specified epochs and logs losses.
        
        Parameters:
            epochs (int): Total number of training epochs
        """
        step = 0

        for epoch in range(epochs):
            self.model.train()

            # ----------
            # Run Training Epoch
            # ----------
            epoch_loss = 0.0
            num_batches = 0

            for x, _ in tqdm(self.dataloader, desc=f"Epoch {epoch}", unit="Batch"):
                # ----------
                # Perform Train Step
                # ----------
                x = x.to(self.device)
                loss = self.train_step(x)

                epoch_loss += loss.item()
                num_batches += 1
                step += 1

                # ----------
                # Log Batch Loss
                # ----------
                if step > 0 and step % self.log_interval == 0:
                    self.log_loss("train/batch_loss", loss.item(), step, epoch)

                # ----------
                # Save Checkpoint
                # ----------
                if step > 0 and step % self.ckpt_interval == 0:
                    self.save_checkpoint(step)
                
                # ----------
                # Log Samples / Sample Stats
                # ----------
                if step > 0 and step % self.sample_interval == 0:
                    frames = self.diffusion.sample_frames(self.ema_model, self.sample_shape, self.num_frames)
                    samples = frames[-1]    # final frame is output sample

                    self.log_sample_stats("sample_stats", samples, step, epoch)

                    video_path = os.path.join(self.save_dir, self.run_name, "figs", f"video-step{step}.mp4")
                    image_path = os.path.join(self.save_dir, self.run_name, "figs", f"grid-step{step}.png")
                    
                    video_frames = make_sample_video(frames, self.fps, video_path)
                    image = make_sample_image(samples, image_path)

                    self.log_video("samples/video", video_frames, step, epoch)
                    self.log_image("samples/image", image, step, epoch)

            # ----------
            # Log Average Epoch Loss
            # ----------
            epoch_loss /= num_batches
            self.log_loss("train/epoch_loss", epoch_loss, step, epoch)

            self.test_sampling()

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a single forward pass / updates the base model and EMA model.
        
        Parameters:
            x (torch.Tensor): Batch of input images of shape (B, C, H, W)
        
        Returns:
            loss (torch.Tensor): Mean squared error loss of predicted noise and true noise
        """
        self.optimizer.zero_grad()

        # ----------
        # Sample Batch of t-values [1,T]
        # ----------
        batch_size = x.shape[0]
        t = torch.randint(1, self.diffusion.T+1, (batch_size,), device=self.device)

        # ----------
        # Apply Noise
        # ----------
        x_t, target = self.diffusion.forward_noise(x, t)

        # ----------
        # Forward Pass
        # ----------
        pred = self.model(x_t, t)

        # ----------
        # Compute Loss / Update
        # ----------
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.optimizer.step()
        self.update_ema()

        return loss
    
    @torch.no_grad()
    def update_ema(self):
        """
        Updates EMA model using exponential moving average of the 
        base model with ema_decay
            - ema_model = decay * ema_model + (1 - decay) * model
        """
        decay = self.ema_decay
        for p_ema, p_model in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.mul_(decay).add_(p_model, alpha=1 - decay)
    
    def log_loss(self, label: str, loss: float, step: int, epoch: int):
        """
        Logs loss to wandb dashboard
        
        Args:
            label (str): Label for metric on dashboard
            loss (float): Current loss to log on dashboard
            step (int): Current training step
        """
        wandb.log(
            {
                label: loss, 
                "epoch": epoch
            }, 
            step=step
        )

    def log_video(self, label: str, video_frames: list[np.ndarray], step: int, epoch: int):
        """
        
        
        Args:
            label (str): Label for grid of samples on dashboard.
            video_frames (list[np.ndarray]): List of frame images of generated samples, each of shape (C, H, W)
            step (int): Current training step
            epoch (int): Current training epoch
        """
        video_frames = np.stack(video_frames, axis=0)           # (T, H, W, C)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2)) # (T, C, H, W) - required wandb.Video shape

        wandb.log(
            {
                label: wandb.Video(video_frames, fps=self.fps, format="mp4"),
                "epoch": epoch
            },
            step=step
        )

    def log_image(self, label: str, image: np.ndarray, step: int, epoch: int):
        """
        Logs image of generated samples grid to wandb dashboard.
        
        Args:
            label (str): Label for grid of samples on dashboard.
            image (np.ndarray): Image of grid of generated samples of shape (C, H, W)
            step (int): Current training step
        """
        wandb.log(
            {
                label: wandb.Image(image), 
                "epoch": epoch
            }, 
            step=step
        )

    def log_sample_stats(self, label: str, samples: torch.Tensor, step: int, epoch: int):
        """
        Logs mean/std of generated samples to wandb dashboard.
        
        Args:
            label (str): Label for grid of samples on dashboard.
            samples (torch.Tensor): Batch of generated samples of shape (B, C, H, W)
            step (int): Current training step
            epoch (int): Current training epoch
        """
        s_mean = samples.mean().item()
        s_std  = samples.std().item()
        
        wandb.log(
            {
                label + "/mean": s_mean,
                label + "/std": s_std,
                "epoch": epoch
            },
            step=step
        )

    def save_checkpoint(self, step: int):
        """
        Saves model checkpoint within save_dir/checkpoints.

        Args:
            step (int): Current training step
        """
        ckpt_path = os.path.join(self.save_dir, self.run_name, "checkpoints", f"model-step{step}.ckpt")
        torch.save({
            "model": self.model.state_dict(), 
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step
        }, ckpt_path)

        artifact = wandb.Artifact(
            name=f"model-step{step}",
            type="model"
        )
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
