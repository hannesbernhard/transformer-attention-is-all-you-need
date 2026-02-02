import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) / rank)
        self.scaling = 1.0  # fully fine-tuned model, 0=base model, 1=LoRA model

    def forward(self, x):
        return self.scaling * (x @ self.lora_A @ self.lora_B)


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()
        self.base_layer = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, linear_layer.out_features, rank=rank
        )

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out


class LoRATransformerModel(nn.Module):
    def __init__(self, base_model, rank=4):
        super().__init__()
        self.base_model = base_model

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Apply LoRA to attention layers in encoder
        for layer in self.base_model.encoder.encoder_layers:
            layer.self_attention.query_transform = LoRALinear(
                layer.self_attention.query_transform, rank
            )
            layer.self_attention.key_transform = LoRALinear(
                layer.self_attention.key_transform, rank
            )
            layer.self_attention.value_transform = LoRALinear(
                layer.self_attention.value_transform, rank
            )

        # Apply LoRA to attention layers in decoder
        for layer in self.base_model.decoder.decoder_layers:
            layer.self_attention.query_transform = LoRALinear(
                layer.self_attention.query_transform, rank
            )
            layer.self_attention.key_transform = LoRALinear(
                layer.self_attention.key_transform, rank
            )
            layer.self_attention.value_transform = LoRALinear(
                layer.self_attention.value_transform, rank
            )
            layer.encoder_attention.query_transform = LoRALinear(
                layer.encoder_attention.query_transform, rank
            )
            layer.encoder_attention.key_transform = LoRALinear(
                layer.encoder_attention.key_transform, rank
            )
            layer.encoder_attention.value_transform = LoRALinear(
                layer.encoder_attention.value_transform, rank
            )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.base_model(src, tgt, src_mask, tgt_mask)


# Usage example:
def setup_lora_model(base_model, rank=4):
    # Load base model
    base_model.eval()

    # Create LoRA model
    lora_model = LoRATransformerModel(base_model, rank=rank)
    return lora_model
