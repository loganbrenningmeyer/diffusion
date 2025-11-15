import torch
import numpy as np

from diffusion.models.unet import UNet


class Diffusion:
    """
    
    Args:
        T (int): 
        beta_1 (float): 
        beta_T (float): 
        beta_schedule (float): 
        sampler (str): 
        ddim_steps (int): 
        eta (float): 
        device (torch.device): 
    """
    def __init__(
            self, 
            T: int, 
            beta_1: float, 
            beta_T: float, 
            beta_schedule: str, 
            sampler: str,
            ddim_steps: int,
            eta: float,
            device: torch.device
    ):
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.beta_schedule = beta_schedule
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.device = device

        # ----------
        # Define betas: (T+1,) - [beta_0, beta_T]
        # => \beta_{0:T} = \{\beta_0,\beta_1,\beta_2,\ldots,\beta_T\},\quad \beta_0 = 0
        # ----------
        betas = self._make_betas()
        self.betas = torch.cat([torch.tensor([0.0], device=device), betas]) 

        # ----------
        # Define alphas: (T+1,) - [alpha_0, alpha_T] 
        # => \alpha_t = 1 - \beta_t
        # => \alpha_{0:T} = \{\alpha_0,\alpha_1,\alpha_2,\ldots,\alpha_T\},\quad \alpha_0 = 1
        # ----------
        alphas = 1.0 - betas
        self.alphas = torch.cat([torch.tensor([1.0], device=device), alphas])

        # ----------
        # Define alpha_bars: (T+1,) - [alpha_bar_0, alpha_bar_T]
        # => \bar\alpha_t = \prod_{s=1}^t \alpha_s
        # => \bar\alpha_{0:T} = \{\bar\alpha_0,\bar\alpha_1,\bar\alpha_2,\ldots,\bar\alpha_T\},\quad \bar\alpha_0 = 1
        # ----------
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.alpha_bars = torch.cat([torch.tensor([1.0], device=device), alpha_bars])

    def forward_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Noises clean input image x_0 to the given timestep t
        
        Args:
            x_0 (torch.Tensor): Batch of clean input images of shape (B, C, H, W)
            t (torch.Tensor): Tensor of timesteps in [1,T] of shape (B,)
        
        Returns:
            x_t (torch.Tensor): Batch of noised images of shape (B, C, H, W)
            eps (torch.Tensor): Batch of added Gaussian noise of shape (B, C, H, W)
        """
        # ----------
        # Sample Gaussian noise
        # => \epsilon \sim \mathcal{N}(0,I)
        # ----------
        eps = torch.randn_like(x_0)

        # ----------
        # Compute xt using closed-form derivation
        # => q(x_t | x_0) = \mathcal{N}(x_t;\ \sqrt{\bar\alpha_t} x_0,\ (1 - \bar\alpha_t)I)
        # => x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon
        # ----------
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

        return x_t, eps
    
    def sample_image(self, model: UNet, shape: tuple[int]) -> torch.Tensor:
        """
        Coordinates generating a batch of image samples of the specified
        shape (B,C,H,W) using either DDPM or DDIM sampling.
        
        Args:
            model (UNet): Trained DDPM UNet for noise prediction
            shape (tuple[int]): Shape of generated samples (B, C, H, W)
        
        Returns:
            x_t (torch.Tensor): Batch of generated samples of shape (B, C, H, W)
        """
        # ----------
        # DDPM
        # ----------
        if self.sampler == "ddpm":
            return self.sample_ddpm_image(model, shape)
        
        # ----------
        # DDIM
        # ----------
        elif self.sampler == "ddim":
            return self.sample_ddim_image(model, shape)
        
    def sample_frames(self, model: UNet, shape: tuple[int], num_frames: int) -> torch.Tensor:
        """
        
        
        Args:
            model (UNet): 
            shape: tuple[int]: 
            num_frames (int): 
        
        Returns:
            x_ts (list[torch.Tensor]): 
        """
        # ----------
        # DDPM
        # ----------
        if self.sampler == "ddpm":
            return self.sample_ddpm_frames(model, shape, num_frames)

        # ----------
        # DDIM
        # ----------
        elif self.sampler == "ddim":
            return self.sample_ddim_frames(model, shape, num_frames)

    @torch.no_grad()
    def sample_ddpm_image(self, model: UNet, shape: tuple[int]) -> torch.Tensor:
        """
        Performs DDPM sampling to generate a batch of samples of the specified 
        shape (B,C,H,W) using the diffusion UNet.
        
        Args:
            model (UNet): Trained DDPM UNet for noise prediction
            shape (tuple[int]): Shape of generated samples (B, C, H, W)
        
        Returns:
            x_t (torch.Tensor): Batch of generated samples of shape (B, C, H, W)
        """
        # ----------
        # Sample Gaussian Noise (x_T)
        # ----------
        x_t = torch.randn(shape, device=self.device)

        # ----------
        # Iteratively Denoise t = T -> 1
        # ----------
        for t_i in range(self.T, 0, -1):
            # ----------
            # Create t batch Tensor
            # ----------
            t = torch.full((shape[0],), t_i, device=self.device, dtype=torch.long)

            # ----------
            # Perform DDPM step
            # ----------
            x_t = self._ddpm_step(model, x_t, t)

        return x_t
    
    @torch.no_grad()
    def sample_ddpm_frames(self, model: UNet, shape: tuple[int], num_frames: int) -> list[torch.Tensor]:
        """
        
        
        Args:
            model (UNet): 
            shape: tuple[int]: 
            num_frames (int): 
        
        Returns:
            frames (list[torch.Tensor]): 
        """
        # ----------
        # Sample Gaussian Noise (x_T)
        # ----------
        x_t = torch.randn(shape, device=self.device)

        # ----------
        # Init Frames List / Define Frame Steps
        # ----------
        frames = [x_t.cpu()]
        frame_ts = set([round(1 + i * (self.T - 1) / (num_frames - 1)) for i in range(num_frames)][:-1])     # {1...T} \ T

        # ----------
        # Iteratively Denoise: t = [T...1]
        # ----------
        for t_i in range(self.T, 0, -1):
            # ----------
            # Create t batch Tensor
            # ----------
            t = torch.full((shape[0],), t_i, device=self.device, dtype=torch.long)

            # ----------
            # Perform DDPM step: x_t -> x_t-1
            # ----------
            x_t = self._ddpm_step(model, x_t, t)

            # ----------
            # Store Frame
            # ----------
            if t_i in frame_ts:
                frames.append(x_t.cpu())

        return frames
    
    @torch.no_grad()
    def sample_ddim_image(self, model: UNet, shape: tuple[int]) -> torch.Tensor:
        """
        Performs DDIM sampling to generate a batch of samples of the specified
        shape (B,C,H,W) using the diffusion UNet.
        
        Args:
            model (UNet): Trained DDPM UNet for noise prediction
            shape (tuple[int]): Shape of generated samples (B, C, H, W)
        
        Returns:
            x_t (torch.Tensor): Batch of generated samples of shape (B, C, H, W)
        """
        # ----------
        # Sample Gaussian Noise (x_T)
        # ----------
        x_t = torch.randn(shape, device=self.device)

        # ----------
        # Uniformly space ddim_steps t-values [T...1] + [0] for t_prev at t_i = 1
        # => t_i = \text{round}\Bigl(T-(i-1)\frac{T-1}{S-1}\Bigr),\quad i=1,\ldots,S
        # ----------
        t_vals = self._make_ddim_timesteps()

        # ----------
        # Iteratively Denoise t = T->1
        # ----------
        for i, t_i in enumerate(t_vals):
            # ----------
            # Create t / t_prev batch Tensors
            # ----------
            t = torch.full((shape[0],), t_i, device=self.device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_vals[i+1], device=self.device, dtype=torch.long)

            # ----------
            # Perform DDIM Step
            # ----------
            x_t, x_0 = self._ddim_step(model, x_t, t, t_prev)
                   
            # ----------
            # At t_i = 1, return x_0
            # ----------
            if t_i == 1:
                x_t = x_0
                break

        return x_t
    
    @torch.no_grad()
    def sample_ddim_frames(self, model: UNet, shape: tuple[int], num_frames: int) -> list[torch.Tensor]:
        """
        
        
        Args:
            model (UNet): 
            shape: tuple[int]: 
            num_frames (int): 
        
        Returns:
            frames (list[torch.Tensor]): 
        """
        # ----------
        # Sample Gaussian Noise (x_T)
        # ----------
        x_t = torch.randn(shape, device=self.device)

        # ----------
        # Uniformly space ddim_steps t-values [T...1] + [0] for t_prev at t_i = 1
        # => t_i = \text{round}\Bigl(T-(i-1)\frac{T-1}{S-1}\Bigr),\quad i=1,\ldots,S
        # ----------
        t_vals = self._make_ddim_timesteps()

        # ----------
        # Init Frames List / Define Frame Steps
        # ----------
        frames = [x_t.cpu()]
        indices = [round(i * (self.ddim_steps - 1) / (num_frames - 1)) for i in range(num_frames)]
        frame_ts = set([t_vals[i] for i in indices][1:])   # {1...T} \ T

        # ----------
        # Iteratively Denoise t = T->1
        # ----------
        for i, t_i in enumerate(t_vals):
            # ----------
            # Create t / t_prev batch Tensors
            # ----------
            t = torch.full((shape[0],), t_i, device=self.device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_vals[i+1], device=self.device, dtype=torch.long)

            # ----------
            # Perform DDIM Step
            # ----------
            x_t, x_0 = self._ddim_step(model, x_t, t, t_prev)

            # ----------
            # Store Frame
            # ----------
            if t_i in frame_ts:
                # ----------
                # At t_i = 1, return x_0
                # ----------
                if t_i == 1:
                    frames.append(x_0.cpu())
                    break
                else:
                    frames.append(x_t.cpu())

        return frames

    def _make_ddim_timesteps(self) -> list[int]:
        """
        Returns a list of evenly spaced DDIM timesteps from T to 1, inclusive,
        with 0 appended for compatibility with t_prev during sampling.
        
        Returns:
            t_vals (list[int]): 
        """
        return [round(self.T - i * (self.T - 1) / (self.ddim_steps - 1)) for i in range(self.ddim_steps)] + [0]

    @torch.no_grad()
    def _ddpm_step(self, model: UNet, x_t: torch.Tensor, t: torch.Tensor):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # ----------
        # Forward Pass
        # ----------
        eps_theta = model(x_t, t)

        # ----------
        # Compute estimated posterior mean
        # => \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Bigr)
        # ----------
        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]

        mu_theta = 1.0/torch.sqrt(alpha_t) * (x_t - (beta_t/torch.sqrt(1 - alpha_bar_t)) * eps_theta)

        # ----------
        # Compute posterior variance
        # => \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t
        # ----------
        alpha_bar_tm1 = self.alpha_bars[t - 1][:, None, None, None]

        beta_tilde = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t

        # ----------
        # Sample Gaussian Noise / Compute x_t-1
        # => x_{t-1} = \mu_\theta(x_t,t) + \sqrt{\tilde\beta_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)\quad \text{if}\ t > 1\ \text{else}\ \epsilon=0
        # ----------
        t_i = t[0].item()
        eps = torch.randn_like(x_t) if t_i > 1 else torch.zeros_like(x_t)

        x_t = mu_theta + torch.sqrt(beta_tilde) * eps

        return x_t

    @torch.no_grad()
    def _ddim_step(self, model: UNet, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # ----------
        # Forward Pass
        # ----------
        eps_theta = model(x_t, t)

        # ----------
        # Compute estimated clean x_0
        # => \hat{x}_0 = \frac{x_{t_i} - \sqrt{1-\bar\alpha_{t_i}}\epsilon_\theta(x_{t_i},{t_i})}{\sqrt{\bar\alpha_{t_i}}}
        # ----------
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]

        x_0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)

        # ----------
        # Compute standard deviation of added noise
        # => \sigma_t(\eta) = \eta \sqrt{\frac{1-\bar\alpha_{t_{i-1}}}{1-\bar\alpha_{t_i}}} \sqrt{1-\frac{\bar\alpha_{t_i}}{\bar\alpha_{t_{i-1}}}}
        # ----------
        alpha_bar_prev = self.alpha_bars[t_prev][:, None, None, None]

        sigma_t = self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)

        # ----------
        # Compute x_t-1
        # => x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2(\eta)}\epsilon_\theta(x_t,t) + \sigma_t(\eta)z,\quad z \sim \mathcal{N}(0,I)
        # ----------
        z = torch.randn_like(x_t) if self.eta != 0 else torch.zeros_like(x_t)

        x_t = torch.sqrt(alpha_bar_prev) * x_0 + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_theta + sigma_t * z

        return x_t, x_0
    
    def _make_betas(self) -> torch.Tensor:
        """
        Creates tensor of beta variance values based on the given
        starting/ending beta values and specified schedule

        Returns:
            betas (torch.Tensor): Tensor of shape (timesteps,) of beta values from t=1->T
        """
        # ----------
        # Linear Schedule
        # ----------
        if self.beta_schedule == "linear":
            # => \beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
            betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)

        # ----------
        # Cosine Schedule
        # ----------
        elif self.beta_schedule == "cosine":
            s = 0.008
            t = torch.arange(0, self.T+1, device=self.device)
            # => f(t) = \cos^2\Bigl(\frac{t/T+s}{1+s} \cdot \frac{\pi}{2}\Bigr)
            f_t = torch.cos((t/self.T + s)/(1 + s) * torch.pi/2)**2
            # => \bar\alpha_t = \frac{f(t)}{f(0)}
            alpha_bars = f_t / f_t[0]
            # => \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}
            betas = 1.0 - alpha_bars[1:]/alpha_bars[0:self.T]
            # -- Clamp betas to avoid 0 and 1
            betas = betas.clamp(min=1e-12, max=0.999)
        
        return betas