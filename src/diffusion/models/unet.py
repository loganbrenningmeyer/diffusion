import torch
import torch.nn as nn
import torchvision


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        # ---------
        # Skip
        # ----------
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        # ---------
        # Time Embedding Projection
        # ----------
        self.t_proj = nn.Linear(t_dim, out_ch)

        # ---------
        # Norms/Convolutions
        # ----------
        self.act   = nn.SiLU()

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch)
        
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch)

    def forward(self, x, t_emb):
        skip = self.skip(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x += self.t_proj(t_emb)[:, :, None, None]
        
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += skip
        x *= (2 ** -0.5)
        
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, down: bool):
        super().__init__()
        # ---------
        # Residual Blocks
        # ----------
        self.res1 = ResBlock(in_ch,  out_ch, t_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_dim)

        # ---------
        # Downsampling
        # ----------
        self.down = Downsample(out_ch) if down else nn.Identity()

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)

        return x, skip


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
    
    def forward(self, x):
        pass


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, up: bool):
        super().__init__()
        


class UNet(nn.Module):
    def __init__(self):
        super().__init__()