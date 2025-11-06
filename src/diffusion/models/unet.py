import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        # ---------
        # Skip Projection
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


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, t_dim, up: bool):
        super().__init__()
        # ---------
        # Upsampling
        # ----------
        self.up   = Upsample(in_ch, out_ch) if up else nn.Identity()
        # ---------
        # Residual Blocks
        # ----------
        res1_in_ch = out_ch + skip_ch if up else out_ch
        self.res1  = ResBlock(res1_in_ch, out_ch, t_dim)
        self.res2  = ResBlock(out_ch,     out_ch, t_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_ch, num_heads):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.attn = nn.MultiheadAttention(in_ch, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        # -- GroupNorm / Flatten
        x_norm = self.norm(x)
        x_flat = x_norm.flatten(2).permute(0,2,1)   # (B, H*W, C)
        # -- Self-attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0,2,1)          # (B, C, H*W)
        attn_out = attn_out.view(b, c, h, w)        # (B, C, H, W)
        return x + attn_out

class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, ch_mults=[1,1,2,2,4,4]):
        super().__init__()
        # ---------
        # Time Embedding MLP
        # ----------
        t_dim = base_ch * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        # ---------
        # Stem
        # ----------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, base_ch),
            nn.SiLU()
        )
        in_ch = base_ch
        # ---------
        # Encoder
        # ----------

        # ---------
        # Bottleneck
        # ----------

        # ---------
        # Decoder
        # ----------

        # ---------
        # Final Layer
        # ----------

    def forward(self, x, t):
        pass