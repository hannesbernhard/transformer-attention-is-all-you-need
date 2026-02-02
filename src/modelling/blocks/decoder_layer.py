import torch
import torch.nn as nn
from src.modelling.layers.multi_head_attention import MultiHeadAttention
from src.modelling.layers.feed_forward import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer using post-layer normalization.

    The layer consists of:
    1. Masked multi-head self-attention (causal / future masking)
    2. Residual connection followed by layer normalization
    3. Multi-head encoder-decoder (cross) attention
    4. Residual connection followed by layer normalization
    5. Position-wise feed-forward network
    6. Residual connection followed by layer normalization

    Dropout is applied after each sublayer.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        feature_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize the Transformer decoder layer.

        Args:
            input_dim (int):
                Dimensionality of the input embeddings (input_dim).
            num_heads (int):
                Number of attention heads.
            feature_dim (int):
                Hidden dimensionality of the feed-forward network (d_ff).
            dropout (float, optional):
                Dropout probability. Default is 0.1.
        """
        super().__init__()

        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)

        # Encoder-decoder (cross) attention
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads)

        # Feed-forward network
        self.feature_transformation = PositionwiseFeedForward(
            input_dim, feature_dim, dropout
        )

        # Independent layer normalizations
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        # Shared dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer decoder layer.

        Args:
            tgt (torch.Tensor):
                Target sequence embeddings of shape (batch_size, tgt_len, input_dim).
            memory (torch.Tensor):
                Encoder output of shape (batch_size, src_len, input_dim).
            tgt_mask (torch.Tensor | None, optional):
                Target mask for masked self-attention (causal + padding).
            memory_mask (torch.Tensor | None, optional):
                Encoder padding mask for cross-attention.

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, tgt_len, input_dim).
        """

        # 1. Masked self-attention
        self_attention_output = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(self_attention_output)
        tgt = self.layer_norm_1(tgt)

        # 2. Encoder-decoder (cross) attention
        cross_attention_output = self.encoder_attention(
            tgt, memory, memory, memory_mask
        )
        tgt = tgt + self.dropout(cross_attention_output)
        tgt = self.layer_norm_2(tgt)

        # 3. Feed-forward network
        ff_output = self.feature_transformation(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.layer_norm_3(tgt)

        return tgt
