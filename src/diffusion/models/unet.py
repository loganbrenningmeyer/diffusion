import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int):
    '''
    Parameters:
    - t: (B,) Integer timesteps
    - dim: Embedding dimension
    '''
    half_dim = dim // 2
    # ----------
    # => k = 0,1,2,\ldots,\frac{d}{2}-1
    # ----------
    k_vals = torch.arange(half_dim, device=t.device)
    # ----------
    # => \omega_k t = \frac{t}{10000^{2k/d}} = \exp(-2k/d \cdot \ln(10000))
    # ----------
    freqs = torch.exp(-2*k_vals/dim * math.log(10000))  # (dim/2,)
    freqs = freqs[None, :] * t[:, None].float()         # (B, dim/2)
    # ----------
    # => \text{PE}(t) = \{\sin(\omega_0 t),\sin(\omega_1 t),\ldots,\sin(\omega_{d/2-1}t),\cos(\omega_0 t),\cos(\omega_1 t),\ldots,\cos(\omega_{d/2-1}t)\}
    # ----------
    t_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)  # (B, dim)
    return t_emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        # ----------
        # Skip Projection
        # ----------
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()
        # ----------
        # Time Embedding Projection
        # ----------
        self.t_proj = nn.Linear(t_dim, out_ch)
        # ----------
        # Norms/Convolutions
        # ----------
        self.act   = nn.SiLU()

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, num_heads, down: bool):
        super().__init__()
        # ----------
        # Residual Blocks
        # ----------
        self.res1 = ResBlock(in_ch, in_ch, t_dim)
        self.res2 = ResBlock(in_ch, in_ch, t_dim)
        # ----------
        # Self-Attention
        # ----------
        self.attn = SelfAttentionBlock(in_ch, num_heads) if num_heads != 0 else nn.Identity()
        # ----------
        # Downsampling
        # ----------
        self.down = Downsample(in_ch, out_ch) if down else nn.Identity()

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
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
    def __init__(self, in_ch, out_ch, skip_ch, t_dim, num_heads, up: bool):
        super().__init__()
        # ----------
        # Upsampling
        # ----------
        self.up = Upsample(in_ch, out_ch) if up else nn.Identity()
        # ----------
        # Residual Blocks
        # ----------
        res1_in_ch = out_ch + skip_ch if up else out_ch
        self.res1  = ResBlock(res1_in_ch, out_ch, t_dim)
        self.res2  = ResBlock(out_ch,     out_ch, t_dim)
        # ----------
        # Self-Attention
        # ----------
        self.attn = SelfAttentionBlock(out_ch, num_heads) if num_heads != 0 else nn.Identity()

    def forward(self, x, t_emb, skip=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) if skip else x
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.attn = nn.MultiheadAttention(in_ch, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        # -- GroupNorm / Flatten
        x_norm = self.norm(x)
        x_flat = x_norm.flatten(2).permute(0,2,1)   # (B, H*W, C)
        # -- Self-Attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0,2,1)          # (B, C, H*W)
        attn_out = attn_out.view(b, c, h, w)        # (B, C, H, W)
        return x + attn_out
    

class Bottleneck(nn.Module):
    def __init__(self, in_ch, t_dim, num_heads=4):
        super().__init__()
        self.res1 = ResBlock(in_ch, in_ch, t_dim)
        self.attn = SelfAttentionBlock(in_ch, num_heads)
        self.res2 = ResBlock(in_ch, in_ch, t_dim)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x
    

class FinalLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.act  = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    '''
    
    '''
    def __init__(self, 
                 in_ch=3, 
                 base_ch=128, 
                 ch_mults=[1,1,2,2,4,4],
                 enc_heads=[0,0,0,0,8,8],
                 mid_heads=4):
        super().__init__()

        self.base_ch = base_ch

        # ----------
        # Time Embedding MLP
        # ----------
        t_dim = base_ch * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )

        # ----------
        # Stem
        # ----------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, base_ch),
            nn.SiLU()
        )
        # -- Update post-stem in_ch
        in_ch = base_ch

        # ----------
        # Encoder
        # ----------
        self.encoder = nn.ModuleList()
        for i, (ch_mult, num_heads) in enumerate(zip(ch_mults, enc_heads)):
            # -- No downsampling at final block
            down = (i != len(ch_mults) - 1)
            # -- Compute out_ch based on block's ch_mult
            out_ch = base_ch*ch_mult
            # -- Initialize EncoderBlock
            self.encoder.append(EncoderBlock(in_ch, out_ch, t_dim, num_heads, down))
            # -- Update in_ch for next EncoderBlock
            in_ch = out_ch

        # ----------
        # Bottleneck
        # ----------
        self.mid = Bottleneck(in_ch, t_dim, mid_heads)

        # ----------
        # Decoder
        # ----------
        self.decoder = nn.ModuleList()
        for i, (ch_mult, num_heads) in enumerate(zip(ch_mults[::-1], enc_heads[::-1])):
            # -- No upsampling at first block
            up = (i != 0)
            # -- Compute out_ch and skip_ch based on block's ch_mult
            out_ch = skip_ch = base_ch*ch_mult
            # -- Initialize DecoderBlock
            self.decoder.append(DecoderBlock(in_ch, out_ch, skip_ch, t_dim, num_heads, up))
            # -- Update in_ch for next DecoderBlock
            in_ch = out_ch

        # ----------
        # Final Layer
        # ----------
        self.final = FinalLayer(in_ch, 3)

    def forward(self, x, t):
        # ----------
        # Time Embedding
        # ----------
        t_emb = sinusoidal_embedding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # ----------
        # Stem
        # ----------
        x = self.stem(x)

        # ----------
        # Encoder
        # ----------
        skips = []

        for i, enc_block in enumerate(self.encoder):
            x, skip = enc_block(x, t_emb)
            # -- Store skip connections when downsampling (not last block)
            if i != len(self.encoder) - 1:
                skips.append(skip)

        # ----------
        # Bottleneck
        # ----------
        x = self.mid(x, t_emb)

        # ----------
        # Decoder
        # ----------
        for i, dec_block in enumerate(self.decoder):
            # -- Concatenate skip connections when upsampling (not first block)
            skip = skips.pop() if i != 0 else None
            x = dec_block(x, t_emb, skip)

        # ----------
        # Final Layer
        # ----------
        x = self.final(x)

        return x