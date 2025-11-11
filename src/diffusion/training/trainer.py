import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
import wandb
from tqdm import tqdm

from diffusion.models.unet import UNet
from diffusion.diffusion import Diffusion


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
    """
    def __init__(
            self, 
            model: UNet, 
            diffusion: Diffusion, 
            optimizer: Optimizer, 
            dataloader: DataLoader, 
            device: torch.device, 
            train_config: DictConfig
    ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.ema_decay = train_config.ema_decay
        self.log_interval = train_config.log_interval
        self.save_interval = train_config.save_interval

    def train(self, epochs: int):
        """
        Trains the UNet model for the specified epochs and logs losses.
        
        Parameters:
            epochs (int): Total number of training epochs
        """
        for epoch in range(epochs):
            # ----------
            # Run Training Epoch
            # ----------
            epoch_loss = 0.0
            num_batches = 0

            for x, _ in tqdm(self.dataloader, desc=f"Epoch {epoch}", unit="Batch"):
                x = x.to(self.device)
                loss = self.train_step(x)

                epoch_loss += loss.item()
                num_batches += 1

                # ----------
                # Log Batch Loss / Save Checkpoint
                # ----------
                if num_batches > 0 and num_batches % self.log_interval == 0:
                    self.log_loss("train/batch_loss", loss.item())

                if num_batches > 0 and num_batches % self.save_interval == 0:
                    self.save_checkpoint()

            # ----------
            # Log Average Epoch Loss
            # ----------
            epoch_loss /= num_batches
            self.log_loss("train/epoch_loss", epoch_loss)

                
    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a single forward pass / update on the input batch
        
        Parameters:
            x (Tensor): Batch of input images of shape (B, C, H, W)
        
        Returns:
            loss (Tensor): Mean squared error loss of predicted noise and true noise
        """
        self.optimizer.zero_grad()

        # ----------
        # Sample Batch of t-values [0, T-1]
        # ----------
        batch_size = x.shape[0]
        t = torch.randint(0, self.diffusion.T, (batch_size,), device=self.device)

        # ----------
        # Apply Noise
        # ----------
        x_t, eps_true = self.diffusion.forward_noise(x, t)

        # ----------
        # Forward Pass
        # ----------
        eps_pred = self.model(x_t, t)

        # ----------
        # Compute Loss / Update
        # ----------
        loss = F.mse_loss(eps_pred, eps_true)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def log_loss(self, label: str, loss: float):
        """
        Logs loss to wandb dashboard.
        
        Args:
            label (str): Label for metric on dashboard.
            loss (float): Current loss to log on dashboard. 
        """
        wandb.log({label: loss})

    def save_checkpoint(self):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        pass