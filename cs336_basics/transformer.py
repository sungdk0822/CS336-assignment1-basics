import torch
from torch import nn
from torch.nn.init import trunc_normal_
from einops import einsum
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        '''
        Construct a linear transformation module. 
        This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        sigma = (2 / (in_features + out_features)) ** 0.5
        self.W = nn.Parameter(
            trunc_normal_(
                torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply the linear transformation to the input.'''
        return einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        Construct an embedding module. This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.embedding = nn.Parameter(
            trunc_normal_(
                torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''Lookup the embedding vectors for the given token IDs.'''
        return self.embedding[token_ids]