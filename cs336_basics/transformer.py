from turtle import forward
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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype, requires_grad=True))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.'''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = ( (x ** 2).sum(dim=-1) / self.d_model + self.eps ) ** 0.5
        result = x * self.g.reshape(1, 1, self.d_model) / rms.unsqueeze_(dim=-1)

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # d_ff = round( (d_model * 8 / 3) / 64 ) * 64
        # if d_ff == 0:
        #     d_ff = 64
        sigma = ( 2 / (d_model + d_ff) ) ** 0.5
        self.W_1 = nn.Parameter(
            trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )
        )
        self.W_2 = nn.Parameter(
            trunc_normal_(
                torch.empty(d_model, d_ff, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )
        )
        self.W_3 = nn.Parameter(
            trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def SiLU(x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)
        return ( SiLU(x @ self.W_1.T) * (x @ self.W_3.T) ) @ self.W_2.T