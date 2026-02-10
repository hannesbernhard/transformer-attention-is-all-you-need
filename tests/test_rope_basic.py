import torch
from src.modelling.layers.rope import get_rope_sin_cos, apply_rope

def test_rope_changes_with_position():
    torch.manual_seed(0)

    B, H, T, D = 1, 1, 8, 8
    x = torch.ones(B, H, T, D)  # identische Tokens an allen Positionen

    sin, cos = get_rope_sin_cos(seq_len=T, head_dim=D, device=x.device)
    x_rope = apply_rope(x, sin, cos)

    # Tokens an verschiedenen Positionen dürfen NICHT gleich sein
    assert not torch.allclose(x_rope[:, :, 0, :], x_rope[:, :, 1, :]), \
        "RoPE failed: positions produce identical embeddings"