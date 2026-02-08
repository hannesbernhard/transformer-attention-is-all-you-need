import math
import torch.nn as nn

from src.modelling.embedding.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    Token embedding + positional encoding
    """

    def __init__(self, vocab_size, d_model, max_len, use_rope = False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope

        if not use_rope:
            self.pos_emb = PositionalEncoding(d_model, max_len)
        else: 
            self.pos_emb = None


    def forward(self, token):
        """
        token: (B, T)
        """
        B, T = token.size()

        token_emb = self.token_emb(token) * math.sqrt(self.token_emb.embedding_dim)

        if self.pos_emb is not None: 
            pos_emb = self.pos_emb(token)             # (B, max_len, d_model) OR similar
            pos_emb = pos_emb[:, :T, :]      # slice to t
            return token_emb + pos_emb     

        return token_emb 
