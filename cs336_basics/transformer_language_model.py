import torch
from torch import inf, nn
from torch.nn.init import trunc_normal_
from einops import einsum

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

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers if needed.
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()

        i_indices = torch.arange(max_seq_len)
        k_indices = torch.arange(0, d_k // 2)
        i_grid, k_grid = torch.meshgrid(i_indices, k_indices)

        theta_values = i_grid / theta ** (2 * k_grid / d_k) # (max_seq_len, d_k // 2)
        cos_values = torch.cos(theta_values) # (max_seq_len, d_k // 2)
        sin_values = torch.sin(theta_values) # (max_seq_len, d_k // 2)

        self.register_buffer(name='cos', tensor=cos_values, persistent=False)
        self.register_buffer(name='sin', tensor=sin_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        You should assume that the token positions are a tensor of shape (..., seq_len) 
        specifying the token positions of x along the sequence dimension.
        '''
        cos_values = self.cos[token_positions] # self.cos: (max_seq_len, d_k // 2) -> cos_values: (seq_len, d_k // 2) # pyright: ignore
        sin_values = self.sin[token_positions] # self.sin: (max_seq_len, d_k // 2) -> sin_values: (seq_len, d_k // 2) # pyright: ignore

        x_even = x[..., ::2] # (batch_size, seq_len, d_k // 2)
        x_odd = x[..., 1::2] # (batch_size, seq_len, d_k // 2)

        rotated_x_even = x_even * cos_values - x_odd * sin_values # (batch_size, seq_len, d_k // 2)
        rotated_x_odd = x_even * sin_values + x_odd * cos_values # (batch_size, seq_len, d_k // 2)

        rotated_x_even.unsqueeze_(dim=-2) # (batch_size, seq_len, 1, d_k // 2)
        rotated_x_odd.unsqueeze_(dim=-2) # (batch_size, seq_len, 1, d_k // 2)

        rotated_x = torch.cat([rotated_x_even, rotated_x_odd], dim=-2) # (batch_size, seq_len, 2, d_k // 2)
        rotated_x = rotated_x.transpose(-2, -1) # (batch_size, seq_len, d_k // 2, 2)
        rotated_x = rotated_x.reshape(*x.shape) # (batch_size, seq_len, d_k)

        return rotated_x

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert -x.ndim <= dim < x.ndim
    if dim < 0:
        dim += x.ndim
    x -= torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

'''
Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)

Deliverable: Implement the scaled dot-product attention function. Your implementation should
handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
(batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
dimensions (if provided). The implementation should return an output with the shape (batch_size,
..., d_v). See section 3.3 for a discussion on batch-like dimensions.
Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
seq_len). The attention probabilities of positions with a mask value of True should collectively sum
to 1, and the attention probabilities of positions with a mask value of False should be zero.
'''
def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask=None):
    '''
    queries: (batch_size, ..., n, d_k)
    keys: (batch_size, ..., m, d_k)
    values: (batch_size, ..., m, d_v)
    mask: (n, m)
    '''
    d_k = queries.shape[-1]
    pre_softmax = queries @ keys.transpose(-2, -1) / d_k ** 0.5 # (batch_size, ..., n, m)
    if mask is not None:
        pre_softmax.masked_fill_(~mask, -inf) # (batch_size, ..., n, m)
    post_softmax = softmax(pre_softmax, dim=-1) # (batch_size, ..., n, m)

    return post_softmax @ values # (batch_size, ..., n, d_v)