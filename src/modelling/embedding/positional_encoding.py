import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
        Computation of sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int):
        """
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """

        super().__init__()

        pos_encoding_matrix = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # each row corresponds to token position

        div_term = torch.exp( # Using torch.exp since it is heavily optimized on GPUs and more stable
            # torch.arange(0, d_model, 2, dtype=torch.float) = (0,2,4,....,d_model) = 2i
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model) # exp(-log(10000)*(2i/d_model)) = 10000**(-2i/d_model) = 1/10000**(2i/d_model)
        )

        pos_encoding_matrix[:, 0::2] = torch.sin(position * div_term) # apply sinus to even dimensions
        pos_encoding_matrix[:, 1::2] = torch.cos(position * div_term) # apply cosinus to odd dimensions

        pos_encoding_matrix = pos_encoding_matrix.unsqueeze(0)

        # Makes tensor part of the model, moves it .to(device) and saves in state_dict
        self.register_buffer("pos_encoding_matrix", pos_encoding_matrix)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the sinusoidal positional encodings for the given input shape.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            Positional encodings of shape (batch_size, sequence_length, d_model)
        """
        seq_len = x.size(1)
        return self.pos_encoding_matrix[:, :seq_len, :]
