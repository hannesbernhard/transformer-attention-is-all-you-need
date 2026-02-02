import torch.nn as nn

from src.modelling.embedding.transformer_embedding import TransformerEmbedding
from src.modelling.blocks.decoder_layer import TransformerDecoderLayer


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, max_len, n_heads, dim_feedforward, dropout, n_layers
    ):
        super().__init__()
        # self.transformer_emb = TransformerEmbedding(vocab_size, d_model, max_len)
        # embedded in Transformer module for consitency in weight sharing
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_emb, tgt_emb, src_mask, tgt_mask):
        # tgt_emb = self.transformer_emb(tgt)

        for layer in self.decoder_layers:
            tgt_emb = layer(src_emb, tgt_emb, src_mask, tgt_mask)
        return tgt_emb
