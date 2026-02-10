import torch
from src.modelling.layers.multi_head_attention import MultiHeadAttention

def test_rope_breaks_permutation_invariance():
    torch.manual_seed(0)

    B, T, D, H = 1, 6, 32, 4
    x = torch.randn(B, T, D)

    perm = torch.tensor([2, 1, 4, 0, 5, 3])
    x_perm = x[:, perm, :]

    mha_no_rope = MultiHeadAttention(D, H, use_rope=False)
    mha_rope = MultiHeadAttention(D, H, use_rope=True)

    out_no_rope = mha_no_rope(x, x, x)
    out_no_rope_perm = mha_no_rope(x_perm, x_perm, x_perm)

    out_rope = mha_rope(x, x, x)
    out_rope_perm = mha_rope(x_perm, x_perm, x_perm)

    # Ohne RoPE: gleiche Werte (bis auf numerisches Rauschen)
    assert torch.allclose(out_no_rope[:, perm], out_no_rope_perm, atol=1e-5)

    # Mit RoPE: MUSS unterschiedlich sein
    assert not torch.allclose(out_rope[:, perm], out_rope_perm, atol=1e-5)
