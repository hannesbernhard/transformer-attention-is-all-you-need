import torch
import torch.nn as nn


class SoftPromptingLayer(nn.Module):
    def __init__(self, prompt_length, embedding_dim):
        super().__init__()
        self.src_soft_prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        self.tgt_soft_prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim))

    def forward(self, src_emb, tgt_emb):
        batch_size = src_emb.size(0)
        src_soft_prompt_expanded = self.src_soft_prompt.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        tgt_soft_prompt_expanded = self.tgt_soft_prompt.unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        return (
            torch.cat([src_soft_prompt_expanded, src_emb], dim=1),
            torch.cat([tgt_soft_prompt_expanded, tgt_emb], dim=1),
        )


class SoftPromptingTransformerModel(nn.Module):
    def __init__(self, base_model, prompt_length, embedding_dim):
        super().__init__()
        self.base_model = base_model
        self.soft_prompting_layer = SoftPromptingLayer(prompt_length, embedding_dim)
        self.src_embedding_layer = base_model.src_embed
        self.tgt_embedding_layer = base_model.tgt_embed
        self.prompt_length = prompt_length

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _adjust_masks(self, original_mask, prompt_length):
        if original_mask is None:
            return None

        batch_size = original_mask.size(0)
        prompt_mask = torch.ones(
            batch_size,
            prompt_length,
            device=original_mask.device,
            dtype=original_mask.dtype,
        )
        return torch.cat([prompt_mask, original_mask], dim=1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed the input tokens using the base model's embedding layers
        src_emb = self.src_embedding_layer(src)
        tgt_emb = self.tgt_embedding_layer(tgt)

        # Add soft prompts to both source and target embeddings
        src_with_prompt, tgt_with_prompt = self.soft_prompting_layer(src_emb, tgt_emb)

        # Adjust masks for prompt tokens
        adjusted_src_mask = self._adjust_masks(src_mask, self.prompt_length)
        adjusted_tgt_mask = self._adjust_masks(tgt_mask, self.prompt_length)

        # Bypass the base model's forward and directly use encoder/decoder
        encoder_out = self.base_model.encoder(src_with_prompt, adjusted_src_mask)
        decoder_out = self.base_model.decoder(
            encoder_out, tgt_with_prompt, adjusted_src_mask, adjusted_tgt_mask
        )

        # Slice off the prompt predictions while keeping original sequence
        output = self.base_model.output_layer(
            decoder_out[:, self.prompt_length - 1 : -1, :]
        )

        return output


def setup_soft_prompting_model(base_model, prompt_length, embedding_dim):
    base_model.eval()
    return SoftPromptingTransformerModel(
        base_model, prompt_length=prompt_length, embedding_dim=embedding_dim
    )
