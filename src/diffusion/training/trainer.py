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

from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion
from diffusion.utils.visualization import make_sample_image, make_sample_video


class Trainer:
    def __init__(
            self, 
            model: UNet, 
            diffusion: Diffusion, 
            optimizer: Optimizer, 
            dataloader: DataLoader, 
            device: torch.device,
            train_dir: str,
            logging_config: DictConfig,
            ema_decay: float
    ):
        """
        Conducts the training process for a diffusion UNet model.

        Provides a `train()` loop and `train_step()` method for 
        forward noising and optimizing the MSE objective.
        
        Args:
            model (UNet): The UNet model used for noise prediction / training.
            diffusion (Diffusion): The Diffusion utilities object for handling diffusion functionality.
            optimizer (Optimizer): The optimizer used for training the model.
            dataloader (DataLoader): Training dataset DataLoader providing batches of images.
            device (torch.device): Device on which training will be performed.
            train_dir (str): Output directory for training run
            logging_config (DictConfig): Config object containing wandb logging / saving parameters
            ema_decay (float): Exponential moving average decay rate
        """
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.train_dir = train_dir

        # -- Logging Parameters
        self.wandb_enabled = logging_config.wandb.enable
        self.wandb_save_ckpt = logging_config.wandb.save_ckpt

        self.loss_interval = logging_config.loss_interval
        self.ckpt_interval = logging_config.ckpt_interval
        self.sample_interval = logging_config.sample_interval

        # ----------
        # Initialize EMA Model
        # ----------
        self.ema_decay = ema_decay

        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    def train(self, steps: int):
        """
        Trains the UNet model for the specified number of steps.
        
        Parameters:
            steps (int): Total number of training steps
        """
        step = 1
        epoch = 1

        while step < steps:
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

                # ----------
                # Log / Save Loss, Samples, and Checkpoint
                # ----------
                self.log_and_save(loss.item(), step, epoch)

                epoch_loss += loss.item()
                num_batches += 1
                step += 1

            # ----------
            # Log Average Epoch Loss
            # ----------
            epoch_loss /= num_batches
            self.log_loss("train/epoch_loss", epoch_loss, step, epoch)

            epoch += 1

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

    def log_and_save(self, loss: float, step: int, epoch: int):
        """
        Logs batch loss, logs/saves samples, and saves checkpoint 
        if at the specified step count
        """
        # ----------
        # Log Batch Loss
        # ----------
        if step > 0 and step % self.loss_interval == 0:
            self.log_loss("train/batch_loss", loss, step, epoch)

        # ----------
        # Log Generated Samples
        # ----------
        if step > 0 and step % self.sample_interval == 0:
            self.save_and_log_samples(step, epoch)

        # ----------
        # Save Checkpoint
        # ----------
        if step > 0 and step % self.ckpt_interval == 0:
            self.save_and_log_checkpoint(step)
    
    def log_loss(self, label: str, loss: float, step: int, epoch: int):
        """
        Logs loss to wandb dashboard
        """
        if self.wandb_enabled:
            wandb.log(
                {
                    label: loss, 
                    "epoch": epoch
                }, 
                step=step
            )

    def save_and_log_samples(self, step: int, epoch: int):
        """
        Uses EMA model to sample, saves to disk, and logs to wandb
        """
        # ----------
        # Generate / Save Sample Trajectories
        # ----------
        frames = self.diffusion.sample_frames(self.ema_model)
        samples = frames[-1]    # final frame is output sample

        video_path = os.path.join(self.train_dir, "figs", f"trajectory-step{step}.gif")
        image_path = os.path.join(self.train_dir, "figs", f"samples-step{step}.png")

        video_frames = make_sample_video(frames, video_path)
        image = make_sample_image(samples, image_path)

        # ----------
        # Log Trajectories Video / Output Samples
        # ----------
        if self.wandb_enabled:
            video_frames = np.stack(video_frames, axis=0)           # (T, H, W, C)
            video_frames = np.transpose(video_frames, (0, 3, 1, 2)) # (T, C, H, W) - required wandb.Video shape

            wandb.log(
                {
                    "figs/trajectory": wandb.Video(video_path, fps=10, format="gif"),
                    "epoch": epoch
                },
                step=step
            )

            wandb.log(
                {
                    "figs/samples": wandb.Image(image_path), 
                    "epoch": epoch
                }, 
                step=step
            )

    def save_and_log_checkpoint(self, step: int):
        """
        Saves model checkpoint at ckpt_path and logs artifact to wandb.
        """
        ckpt_path = os.path.join(self.train_dir, "checkpoints", f"model-step{step}.ckpt")

        torch.save({
            "model": self.model.state_dict(), 
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step
        }, ckpt_path)

        if self.wandb_enabled and self.wandb_save_ckpt:
            artifact = wandb.Artifact(
                name=f"model-step{step}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
