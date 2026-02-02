import torch.nn as nn
import torch

class WordEmbedding(nn.Module):
    """
    This module is a word embedding layer, mapping word indicies to dense vector representations (with a learnable embedding matrix).

    Emb(x) = 1(x) * E
    
    where: 
    - x = tensor of word indicies
    - 1(x) = one hot encoding of x
    - E ∈ R^{vocab_size * d_model} is the embedding matrix
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize the word embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            d_model (int): Dimensionality of each word embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs an embedding lookup for a given tensor.

        Args:
            x (torch.Tensor): Tensor of word indices with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Embedded word representations with shape (batch_size, sequence_length, d_model).
        """
        return self.embedding(x)