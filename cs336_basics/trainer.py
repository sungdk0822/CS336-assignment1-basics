import torch
from collections.abc import Callable
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional


'''Given a tensor of inputs and targets, compute the average cross-entropy
loss across examples.

Args:
    inputs (Float[Tensor, 'batch_size vocab_size']): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, 'batch_size']): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.

Returns:
    Float[Tensor, '']: The average cross-entropy loss across examples.
'''
def cross_entropy(inputs: Float[Tensor, ' batch_size vocab_size'], targets: Int[Tensor, ' batch_size']) -> Float[Tensor, '']:
    stable_inputs = inputs - inputs.max(dim=-1, keepdim=True).values # subtract the largest element for numerical stability
    negative_log_likelihood = stable_inputs.exp().sum(dim=-1).log() - stable_inputs[torch.arange(inputs.shape[0]), targets]
    return negative_log_likelihood.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        weight_decay: float = 0.01, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
    ):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')

        defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'beta_1': betas[0],
            'beta_2': betas[1],
            'eps': eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get('t', 1) # Get iteration number from the state, or initial value.
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                g = p.grad.data # Get the gradient of loss with respect to p.
                
                m = beta_1 * m + (1 - beta_1) * g # update the first moment estimate
                v = beta_2 * v + (1 - beta_2) * g ** 2 # update the second moment estimate
                lr *= (1 - beta_2 ** t) ** 0.5 / (1 - beta_1 ** t) # compute adjusted lr for iteration t

                p.data -= lr * m / (v ** 0.5 + eps) # update the parameters
                p.data *= 1 - group['lr'] * weight_decay # apply weight decay

                state['t'] = t + 1 # Increment iteration number.
                state['m'] = m
                state['v'] = v

        return loss


if __name__ == '__main__':
    pass