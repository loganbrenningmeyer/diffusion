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
    
    return betas


class Diffusion:
    """
    
    """
    def __init__(
            self, 
            T: int, 
            beta_1: float, 
            beta_T: float, 
            beta_schedule: str, 
            device: torch.device
    ):
        self.T = T
        self.betas = create_beta_schedule(T, beta_1, beta_T, beta_schedule, device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.device = device

    def forward_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Noises clean input image x_0 to the given timestep t
        
        Args:
            x_0 (Tensor): Batch of clean input images of shape (B, C, H, W)
            t (Tensor): Tensor of timesteps in [0,T-1] of shape (B,)
        
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
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]   # (B, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

        return x_t, eps

    @torch.no_grad()
    def sample(self, model: UNet) -> torch.Tensor:
        """
        Generates a sample using the UNet from random Gaussian noise by
        denoising from timestep t=T to t=1
        
        Args:
            model (UNet): Trained DDPM UNet
        
        Returns:
            x_0 (Tensor): Sample produced by denoising random Gaussian noise
        """
        pass