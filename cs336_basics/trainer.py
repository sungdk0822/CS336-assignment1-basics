import torch
from torch import Tensor
from jaxtyping import Float, Int


"""Given a tensor of inputs and targets, compute the average cross-entropy
loss across examples.

Args:
    inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.

Returns:
    Float[Tensor, ""]: The average cross-entropy loss across examples.
"""
def cross_entropy(inputs: Float[Tensor, ' batch_size vocab_size'], targets: Int[Tensor, ' batch_size']) -> Float[Tensor, '']:
    stable_inputs = inputs - inputs.max(dim=-1, keepdim=True).values # subtract the largest element for numerical stability
    negative_log_likelihood = stable_inputs.exp().sum(dim=-1).log() - stable_inputs[torch.arange(inputs.shape[0]), targets] # NLL using logsumexp trick
    return negative_log_likelihood.mean()


if __name__ == '__main__':
    pass