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