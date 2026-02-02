import torch
import torch.nn as nn
from src.modelling.layers.multi_head_attention import MultiHeadAttention
from src.modelling.layers.feed_forward import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer using post-layer normalization.
    The layer consits of: 
    1. Multi-head self-attention
    2. Residual connection followed by layer normalization
    3. Position-wise feed-forward network
    4. Residual connection followed by layer normalization

    Dropout is applied after the self-attention and feed-forward sublayers.
    """

    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, dropout: float = 0.1, **kwargs):
        """
        Initialize the Transformer encoder layer.

        Args:
            input_dim (int):
                Dimensionality of the input embeddings (input_dim).
            num_heads (int):
                Number of attention heads in the multi-head attention layer.
            feature_dim (int):
                Hidden dimensionality of the position-wise feed-forward network (d_ff).
            dropout (float, optional):
                Dropout probability applied after attention and feed-forward layers. Defaul is 0.1
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(input_dim, num_heads)
        self.feature_transformation = PositionwiseFeedForward(
            input_dim, feature_dim, dropout
        )

        # Independent layer normalizations for each sublayer
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        # Dropout
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder layer.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            src_mask (torch.Tensor | None, optional): Attention mask indicating valid positions. Shape: (batch_size, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim).
        """
                
        # Self attention
        attn_output = self.self_attention(src, src, src, src_mask)
        src = self.layer_norm_1(src + self.dropout_attn(attn_output))

        # Feed-forward
        ffn_output = self.feature_transformation(src)
        src = self.layer_norm_2(src + self.dropout_ffn(ffn_output))

        return src