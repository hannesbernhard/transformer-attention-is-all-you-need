import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network, applying the same feed-forward network independently
    to each position in the input sequence.

    The transformation is defined as:

        FFN(x) = Linear(d_model → d_ff), expands feature space
                 → ReLU, adds non-linearity
                 → Dropout
                 → Linear(d_ff → d_model) back to original dimension

    Shape:
        Input:  (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the position-wise feed-forward layer.

        Args:
            d_model (int): Dimensionality of the input and output features.
            d_ff (int): Dimensionality of the hidden feed-forward layer.
            dropout (float, optional): Dropout probability applied after
                the activation function. Default is 0.1.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)



    def forward(self, x):
        """
        Forward pass of the feed-forward layer.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, seq_len, d_model).
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x