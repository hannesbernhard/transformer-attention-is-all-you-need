from typing import Optional
import torch
import torch.nn as nn

from src.modelling.layers.attention import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mask_future: bool = False):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(mask_future=mask_future)

        for layer in [
            self.query_transform,
            self.key_transform,
            self.value_transform,
            self.output_transform,
        ]:
            nn.init.xavier_uniform_(layer.weight) # using xavier so that the output values have roughly the same scale as the input values



    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        if mask is not None:
            # mask: (B, K) → (B, 1, 1, K)
            mask = mask.unsqueeze(1).unsqueeze(2)
            
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        # Linear projections
        q = (
            self.query_transform(query)
            .view(batch_size, query_len, self.n_heads, self.d_k) # split each token into n_heads smaller vectors
            .transpose(1, 2)
        ) # Shape: (batch_size, n_heads, seq_len, d_model)
        k = (
            self.key_transform(key)
            .view(batch_size, key_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        ) # Shape: (batch_size, n_heads, seq_len, d_model)
        v = (
            self.value_transform(value)
            .view(batch_size, key_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        ) # Shape: (batch_size, n_heads, seq_len, d_model)

        # Apply attention on all the projected vectors in batch
        attention_output = self.attention(q, k, v, mask)

        # Concatenate heads and put through final linear layer
        attention_output = (
            attention_output.transpose(1, 2) # back to original structure
            .contiguous() # required for .view
            .view(batch_size, query_len, self.d_model) # concatenate to way it was before splitting each token into n_heads vectors
        )

        return self.output_transform(attention_output)