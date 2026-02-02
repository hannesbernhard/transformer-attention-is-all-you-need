import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class Attention(nn.Module): 
  def __init__(self, mask_future=False):
    """
      mask_future (bool): if True, apply future masking
    """
    super().__init__()
    self.mask_future = mask_future


  def forward(
    self,
    query: torch.Tensor, 
    key: torch.Tensor,
    value: torch.Tensor, 
    mask: Optional[torch.Tensor] = None,
  ):
    """
    Computes scaled dot-product attention with optional padding and future masking.

    This implementation is shape-polymorphic and supports both:
    - Single-head attention: (batch_size, seq_len, dim)
    - Multi-head attention:  (batch_size, n_heads, seq_len, dim)

    Args:
        query: Tensor of shape (..., query_len, d_k)
        key:   Tensor of shape (..., key_len,   d_k)
        value: Tensor of shape (..., key_len,   d_v)
        mask:  Optional tensor broadcastable to (..., query_len, key_len), where 1 indicates valid keys and 0 indicates masked positions.

    Returns:
        Tensor of shape (..., query_len, d_v)
    """
    d_k = key.size(-1) # returns last dimension of query tensor
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # key transpose swaps last two dimensions

    # Apply padding mask
    if mask is not None:
      # mask expected shape: (..., key_len) or broadcastable
      while mask.dim() < scores.dim(): # expected initial shape = (B, K) --> after unsqueeze = (B,1,K) or (B,1,1,K) for multi head attention
          mask = mask.unsqueeze(-2)
      mask = mask.bool()
      scores = scores.masked_fill(~mask.bool(), float("-inf"))

    # Apply future mask
    if self.mask_future:
      no_cheating_mask = torch.triu(
          torch.ones(scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool),
          diagonal=1,
        )
      scores = scores.masked_fill(
          no_cheating_mask.view((1,) * (scores.dim() - 2) + no_cheating_mask.shape),
          float("-inf"),
        )

    attention = F.softmax(scores, dim=-1)
    # output: (batch_size, n_heads, query_len, d_v)
    output = torch.matmul(attention, value)
    return output

