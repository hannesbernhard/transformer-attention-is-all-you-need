import torch.nn as nn
from src.modelling.blocks.encoder_layer import TransformerEncoderLayer
from src.modelling.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, max_len, n_heads, dim_feedforward, dropout, n_layers
    ):
        super().__init__()
        # self.transformer_emb = TransformerEmbedding(vocab_size, d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_emb, src_mask):
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)

        return src_emb
