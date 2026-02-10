import torch
from src.modelling.layers.multi_head_attention import MultiHeadAttention

def test_rope_affects_attention_scores():
    torch.manual_seed(0)

    B, T, D, H = 1, 6, 32, 4
    x = torch.randn(B, T, D)

    mha_no_rope = MultiHeadAttention(D, H, use_rope=False)
    mha_rope = MultiHeadAttention(D, H, use_rope=True)

    out_no_rope = mha_no_rope(x, x, x)
    out_rope = mha_rope(x, x, x)

    assert not torch.allclose(out_no_rope, out_rope), \
        "RoPE has no effect on attention output"
