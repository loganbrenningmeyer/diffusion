import torch

from diffusion.models.unet import UNet


def create_beta_schedule(
        T: int,
        beta_1: float, 
        beta_T: float, 
        beta_schedule: str,
        device: torch.device
) -> torch.Tensor:
    """
    Creates tensor of beta variance values based on the given
    starting/ending beta values and specified schedule
    
    Args:
        T (int): Total number of timesteps
        beta_1 (float): Initial beta value in the variance schedule
        beta_T (float): Final beta value in the variance schedule
        beta_schedule (str): Choice of beta scheduling strategy
            - "linear": Uniformly spaces beta values from beta_1 to beta_T
            - "cosine": Defines alpha_bars using a cos^2 decay, then converts to beta values
        device (torch.device): Designated Torch device for betas tensor

    Returns:
        betas (Tensor): Tensor of shape (timesteps,) of beta values from t=1->T
    """
    # ----------
    # Linear Schedule
    # ----------
    if beta_schedule == "linear":
        # => \beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
        betas = torch.linspace(beta_1, beta_T, T, device=device)

    # ----------
    # Cosine Schedule
    # ----------
    elif beta_schedule == "cosine":
        s = 0.008
        t = torch.arange(0, T+1, device=device)
        # => f(t) = \cos^2\Bigl(\frac{t/T+s}{1+s} \cdot \frac{\pi}{2}\Bigr)
        f_t = torch.cos((t/T + s)/(1 + s) * torch.pi/2)**2
        # => \bar\alpha_t = \frac{f(t)}{f(0)}
        alpha_bars = f_t / f_t[0]
        # => \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}
        betas = 1.0 - alpha_bars[1:]/alpha_bars[0:T]
        # -- Clamp betas to avoid 0 and 1
        betas = betas.clamp(min=1e-12, max=0.999)
    
    return betas


class Diffusion:
    """
    
    Args:
        T (int): 
        beta_1 (float): 
        beta_T (float): 
        beta_schedule (float): 
        sampler (str): 
        n_steps (int): 
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
            n_steps: int,
            eta: float,
            device: torch.device
    ):
        self.T = T
        self.sampler = sampler
        self.n_steps = n_steps
        self.eta = eta
        self.device = device

        # ----------
        # Define betas: (T+1,) - [beta_0, beta_T]
        # => \beta_{0:T} = \{\beta_0,\beta_1,\beta_2,\ldots,\beta_T\},\quad \beta_0 = 0
        # ----------
        betas = create_beta_schedule(T, beta_1, beta_T, beta_schedule, device)
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
            x_0 (Tensor): Batch of clean input images of shape (B, C, H, W)
            t (Tensor): Tensor of timesteps in [1,T] of shape (B,)
        
        Returns:
            x_t (Tensor): Batch of noised images of shape (B, C, H, W)
        """
        # ----------
        # Sample Gaussian noise
        # => \epsilon \sim \mathcal{N}(0,I)
        # ----------
        eps = torch.randn_like(x_0, device=self.device)

        # ----------
        # Compute xt using closed-form derivation
        # => q(x_t | x_0) = \mathcal{N}(x_t;\ \sqrt{\bar\alpha_t} x_0,\ (1 - \bar\alpha_t)I)
        # => x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon
        # ----------
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

        return x_t, eps
    
    def sample(self, model: UNet, shape: tuple[int]) -> torch.Tensor:
        """
        Coordinates generating a batch of samples of the specified
        shape (B,C,H,W) using either DDPM or DDIM sampling.
        
        Args:
            model (UNet): Trained DDPM UNet for noise prediction
            shape (tuple[int]): Shape of generated samples (B, C, H, W)
        
        Returns:
            x_t (Tensor): Batch of generated samples of shape (B, C, H, W)
        """
        # ----------
        # DDPM
        # ----------
        if self.sampler == "ddpm":
            return self.sample_ddpm(model, shape)
        
        # ----------
        # DDIM
        # ----------
        elif self.sampler == "ddim":
            return self.sample_ddim(model, shape)

    @torch.no_grad()
    def sample_ddpm(self, model: UNet, shape: tuple[int]) -> torch.Tensor:
        """
        Performs DDPM sampling to generate a batch of samples of the specified 
        shape (B,C,H,W) using the diffusion UNet.
        
        Args:
            model (UNet): Trained DDPM UNet for noise prediction
            shape (tuple[int]): Shape of generated samples (B, C, H, W)
        
        Returns:
            x_t (Tensor): Batch of generated samples of shape (B, C, H, W)
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
            t = torch.full((shape[0],), t_i, device=self.device)

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
            # => x_{t-1} = \mu_\theta(x_t,t) + \sqrt{\tilde\beta_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)\ \text{if}\ t > 1\ \text{else}\ \epsilon=0
            # ----------
            eps = torch.randn_like(x_t, device=self.device) if t_i > 1 else torch.zeros_like(x_t)

            x_t = mu_theta + torch.sqrt(beta_tilde) * eps

        return x_t
    
    @torch.no_grad()
    def sample_ddim(self, model: UNet, shape: tuple[int]) -> torch.Tensor:
        """
        Performs DDIM sampling to generate a batch of samples of the specified
        shape (B,C,H,W) using the diffusion UNet.
        
        Args:
            model (UNet): Trained DDPM UNet for noise prediction
            shape (tuple[int]): Shape of generated samples (B, C, H, W)
        
        Returns:
            x_t (Tensor): Batch of generated samples of shape (B, C, H, W)
        """
        # ----------
        # Sample Gaussian Noise (x_T)
        # ----------
        x_t = torch.randn(shape, device=self.device)

        # ----------
        # Uniformly space n_steps t-values [T...1]
        # => t_i = \text{round}\Bigl(T-(i-1)\frac{T-1}{S-1}\Bigr),\quad i=1,\ldots,S
        # ----------
        t_vals = [round(self.T - (i-1) * (self.T-1)/(self.n_steps-1)) for i in range(1, self.n_steps+1)]

        # ----------
        # Iteratively Denoise t = T->1
        # ----------
        for i, t_i in enumerate(t_vals):
            # ----------
            # Create t batch Tensor
            # ----------
            t = torch.full((shape[0],), t_i, device=self.device)

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
            # At t_i = 1, return x_0
            # ----------
            if t_i == 1:
                x_t = x_0
                break

            # ----------
            # Compute standard deviation of added noise
            # => \sigma_t(\eta) = \eta \sqrt{\frac{1-\bar\alpha_{t_{i-1}}}{1-\bar\alpha_{t_i}}} \sqrt{1-\frac{\bar\alpha_{t_i}}{\bar\alpha_{t_{i-1}}}}
            # ----------
            t_prev = torch.full((shape[0],), t_vals[i+1], device=self.device)
            alpha_bar_prev = self.alpha_bars[t_prev][:, None, None, None]

            sigma_t = self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)

            # ----------
            # Compute x_t-1
            # => x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2(\eta)}\epsilon_\theta(x_t,t) + \sigma_t(\eta)z,\quad z \sim \mathcal{N}(0,I)
            # ----------
            z = torch.randn_like(x_t) if self.eta != 0 else torch.zeros_like(x_t)

            x_t = torch.sqrt(alpha_bar_prev) * x_0 + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_theta + sigma_t * z

        return x_t
