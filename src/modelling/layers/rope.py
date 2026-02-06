import torch
import math


def rotate_half(x):
    """
    Split last dimension into even and odd parts and rotate.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """
    Apply rotary positional embedding.
    x: (batch, heads, seq_len, head_dim)
    """
    return (x * cos) + (rotate_half(x) * sin)


def get_rope_sin_cos(seq_len, head_dim, device):
    """
    Precompute sin and cos for RoPE.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim)
    )

    positions = torch.arange(seq_len, device=device)
    angles = torch.einsum("i,j->ij", positions, inv_freq)

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    # shape: (1, 1, seq_len, head_dim)
    sin = torch.repeat_interleave(sin, 2, dim=-1)[None, None, :, :]
    cos = torch.repeat_interleave(cos, 2, dim=-1)[None, None, :, :]

    return sin, cos
