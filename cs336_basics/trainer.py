import math
import torch
from collections.abc import Callable
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional, Iterable


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


'''
The cosine annealing learning rate schedule takes 
    (i) the current iteration t, 
    (ii) the maximum learning rate alpha_max, 
    (iii) the minimum (final) learning rate alpha_min, 
    (iv) the number of warm-up iterations T_w, and 
    (v) the number of cosine annealing iterations T_c.
'''
'''
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
'''
def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return (
            min_learning_rate 
            + 0.5 
            * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) 
            * (max_learning_rate - min_learning_rate)
        )
    elif it > cosine_cycle_iters:
        return min_learning_rate


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += (parameter.grad.data ** 2).sum()
    l2_norm = total ** 0.5
    if l2_norm > max_l2_norm:
        epsilon = 1e-6
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data *= max_l2_norm / (l2_norm + epsilon)


if __name__ == '__main__':
    g = torch.arange(4).reshape(2,2).float()
    l2_norm = (g ** 2).sum().sqrt().item()
    g *= 0.5
    pass