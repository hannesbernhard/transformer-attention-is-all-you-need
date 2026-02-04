from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modelling.embedding.transformer_embedding import TransformerEmbedding
from src.modelling.blocks.decoder_layer import (
    TransformerDecoderLayer as Decoder,
)
from src.modelling.blocks.encoder_layer  import (
    TransformerEncoderLayer as Encoder,
)


# =========================
# Configuration
# =========================

@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    max_len: int

    def validate(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be positive")
        if self.num_decoder_layers <= 0:
            raise ValueError("num_decoder_layers must be positive")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be between 0 and 1")
        if self.max_len <= 0:
            raise ValueError("max_len must be positive")


# =========================
# Transformer Model
# =========================

class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embeddings
        self.src_embed = TransformerEmbedding(
            config.vocab_size, config.d_model, config.max_len
        )
        self.tgt_embed = TransformerEmbedding(
            config.vocab_size, config.d_model, config.max_len
        )

        # Encoder and Decoder
        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            input_dim=config.d_model,
            max_len=config.max_len,
            num_heads=config.n_heads,
            feature_dim=config.dim_feedforward,
            dropout=config.dropout,
            n_layers=config.num_encoder_layers,
        )

        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            input_dim=config.d_model,
            max_len=config.max_len,
            num_heads=config.n_heads,
            feature_dim=config.dim_feedforward,
            dropout=config.dropout,
            n_layers=config.num_decoder_layers,
        )

        # Output projection
        self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self._tie_weights()

    def _tie_weights(self):
        self.src_embed.token_emb.weight = self.tgt_embed.token_emb.weight
        self.output_layer.weight = self.tgt_embed.token_emb.weight

    # =========================
    # Masks
    # =========================

    @staticmethod
    def _generate_square_subsequent_mask(size: int, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()

    # =========================
    # Forward (training)
    # =========================

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (B, S)
        tgt: (B, T)
        """
        src_emb = self.src_embed(src)
        memory = self.encoder(src_emb, src_mask)

        tgt_emb = self.tgt_embed(tgt)
        #decoder_out = self.decoder(memory, tgt_emb, src_mask, tgt_mask)

        decoder_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.output_layer(decoder_out)
        return logits

    # =========================
    # Greedy Decoding (inference)
    # =========================

    @torch.no_grad()
    def generate(self, src, bos_token_id: int, eos_token_id: int, max_length: int = 50):
        device = src.device
        batch_size = src.size(0)
        assert batch_size == 1, "generate() supports batch_size=1 only"

        # Encode source
        src_emb = self.src_embed(src)
        memory = self.encoder(src_emb)

        # Start token BOS
        tgt = torch.full(
            (1, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )


        for _ in range(max_length):
            tgt_emb = self.tgt_embed(tgt)

            decoder_out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                memory_mask=None,
                tgt_mask=None,  # causal mask handled internally
            )

            logits = self.output_layer(decoder_out)

            # Encourage EOS (decoding-only bias)
            logits[:, -1, eos_token_id] += 1.5

            # repetition penalty
            for t in set(tgt[0].tolist()):
                logits[:, -1, t] /= 1.2

            # encourage EOS after minimum length
            if tgt.size(1) > 8:
                logits[:, -1, eos_token_id] += 2.0

            next_token = logits[:, -1].argmax(dim=-1)

            # probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            if next_token.item() == eos_token_id:
                break

        return tgt