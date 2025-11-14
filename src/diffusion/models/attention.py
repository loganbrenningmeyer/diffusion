import math
import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """
    Applies multi-head self-attention on the batch of inputs x 
    of shape (B, C, H, W) and adds to residual.
    
    Args:
        in_ch (int): Input channel width / embedding dimensionality
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    
    Returns:
        x (Tensor): Self-attended output of shape (B, C, H, W)
    """
    def __init__(
            self, 
            in_ch: int, 
            num_heads: int, 
            dropout: float=0.0
    ):
        super().__init__()

        self.in_ch = in_ch
        self.head_dim = in_ch // num_heads
        self.num_heads = num_heads

        # ----------
        # Query, Key, Value, and Output Projections
        # => W_Q,W_K,W_V,W_O \in \mathcal{R}^{d_\text{model} \times d_\text{model}}
        # ----------
        self.W_q = nn.Linear(in_ch, in_ch)
        self.W_k = nn.Linear(in_ch, in_ch)
        self.W_v = nn.Linear(in_ch, in_ch)
        self.W_o = nn.Linear(in_ch, in_ch)

        # ----------
        # Dropout
        # ----------
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------
        # Spatial Positions (H, W) -> Tokens (T)
        # ----------
        B, C, H, W, = x.shape
        T = H*W

        x = x.flatten(2)        # (B, C, T)
        x = x.transpose(1, 2)   # (B, T, C)

        # ----------
        # Compute Queries, Keys, and Values
        # => x \in \mathcal{R}^{B \times T \times d_\text{model}}
        # => Q = xW_Q,\ K = xW_K,\ V = xW_V,\quad Q,K,V \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        Q: torch.Tensor = self.W_q(x)    # (B, T, C)
        K: torch.Tensor = self.W_k(x)    # (B, T, C)
        V: torch.Tensor = self.W_v(x)    # (B, T, C)

        # ----------
        # Split into heads
        # => Q^{(i)}, K^{(i)}, V^{(i)} \in \mathcal{R}^{B \times T \times d_h},\quad i=1,2,\ldots,h,\quad d_h = \frac{d_\text{model}}{h}
        # ----------
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, T, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, T, head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, T, head_dim)

        # ----------
        # Compute scaled dot-product attention scores
        # => S^{(i)} = \frac{Q^{(i)}K^{(i)\intercal}}{\sqrt{d_\text{h}}} \in \mathcal{R}^{B \times T \times T}
        # ----------
        K_T = K.transpose(2, 3)                 # (B, num_heads, head_dim, T)
        S = Q @ K_T / math.sqrt(self.head_dim)  # (B, num_heads, T, T)

        # ----------
        # Perform row-wise softmax to compute attention weights + Dropout
        # => A^{(i)} = \text{softmax}_\text{row}(S^{(i)}) \in \mathcal{R}^{B \times T \times T}
        # ----------
        A = torch.softmax(S, dim=-1)    # (B, num_heads, T, T)
        A = self.drop(A)

        # ----------
        # Apply attention weights to values to get outputs
        # => Y^{(i)} = A^{(i)}V^{(i)} \in \mathcal{R}^{B \times T \times d_h}
        # ----------
        Y = A @ V   # (B, num_heads, T, head_dim)

        # ----------
        # Concatenate head outputs
        # => Y_\text{concat} = [Y^{(1)};Y^{(2)};\ldots;Y^{(h)}] \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        Y = Y.transpose(1, 2).contiguous()      # (B, T, num_heads, head_dim)
        Y = Y.view(B, T, self.in_ch)            # (B, T, C)

        # ----------
        # Apply output projection
        # => O = Y_\text{concat}W_O \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        O = self.W_o(Y)     # (B, T, C)
        x = x + O

        # ----------
        # Tokens (T) -> Spatial Positions (H, W)
        # ----------
        x = x.transpose(1, 2)   # (B, C, T)
        x = x.view(B, C, H, W)  # (B, C, H, W)

        return x
