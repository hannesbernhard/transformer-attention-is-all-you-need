import torch
import torch.nn.functional as F
from src.modelling.layers.attention import Attention

def test_attention_mechanism():
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Define dimensions and instantiate the Attention module
    d_k = 4
    batch_size = 2
    seq_len = 5
    attention = Attention(mask_future=False)
    # otherwise dropout will be applied during testing

    # Generate random Q, K, V tensors
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    # Compute scaled dot-product attention manually
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    attention_weights = F.softmax(scores, dim=-1)
    expected_output = torch.matmul(attention_weights, V)

    # Get the actual output from the Attention mechanism
    actual_output = attention(Q, K, V)

    # Assert that the expected and actual outputs are close
    assert torch.allclose(
        expected_output, actual_output
    ), "Output mismatch in attention mechanism."