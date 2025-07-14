import math
import numpy as np
import numpy.typing as npt
import os
import torch
import typing
from collections.abc import Callable
from dataclasses import dataclass, fields
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional, Iterable


'''
    Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.

    Args:
        inputs (Float[Tensor, 'batch_size vocab_size']): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, 'batch_size']): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, '']: The average cross-entropy loss across examples.
'''
def cross_entropy(inputs: Float[Tensor, ' batch_size vocab_size'], targets: Int[Tensor, ' batch_size']) -> Float[Tensor, '']:
    if inputs.ndim == 3 and targets.ndim == 2:
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(-1)
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


'''
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
'''
def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]
    input_start_indices = torch.randint(0, n - context_length, (batch_size,))
    input_end_indices = input_start_indices + context_length

    input_start_indices = input_start_indices.tolist()
    input_end_indices = input_end_indices.tolist()

    inputs = [ dataset[start:end] for start, end in zip(input_start_indices, input_end_indices) ]
    labels = [ dataset[start+1:end+1] for start, end in zip(input_start_indices, input_end_indices) ]

    inputs = np.array(inputs) # convert to numpy arrays first to avoid warnings when creating torch tensors
    labels = np.array(labels) # convert to numpy arrays first to avoid warnings when creating torch tensors

    inputs = torch.tensor(inputs, device=device)
    labels = torch.tensor(labels, device=device)

    return inputs, labels


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    entire_state = {
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'iteration': iteration
    }
    torch.save(entire_state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    entire_state = torch.load(src)
    model.load_state_dict(entire_state['model_state'])
    optimizer.load_state_dict(entire_state['optimizer_state'])
    iteration = entire_state['iteration']

    return iteration


'''
    Problem (training_together): Put it together (4 points)
    Deliverable:
    Write a script that runs a training loop to train your model on user-provided input.
    In particular, we recommend that your training script allow for (at least) the following:
    • Ability to configure and control the various model and optimizer hyperparameters.
    • Memory-efficient loading of training and validation large datasets with np.memmap.
    • Serializing checkpoints to a user-provided path.
    • Periodically logging training and validation performance (e.g., to console and/or an external
    service like Weights and Biases).
'''
def train_and_save_tokenizer(
    bpe_train_corpus_path: str,
    vocab_size: int,
    special_tokens: list[str],
    tokenizer_save_path: str
) -> int:
    vocab, merges = train_bpe(bpe_train_corpus_path, vocab_size - 1, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    torch.save(tokenizer, tokenizer_save_path)

    return len(tokenizer.vocab)


def tokenize_and_save_corpus_ids(
    tokenizer_path: str,
    corpus_path: str,
    corpus_ids_save_path: str
) -> None:
    tokenizer = torch.load(tokenizer_path, weights_only=False)
    with open(corpus_path, 'rb') as f:
        corpus = f.read().decode('utf-8', errors='ignore')
        corpus_ids = np.array(tokenizer.encode(corpus))
        np.save(corpus_ids_save_path, corpus_ids)


@dataclass
class CosineLRScheduleConfig:
    it: int = 0
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    warmup_iters: int = 100
    cosine_cycle_iters: int = 1000
    def get_config(self):
        return (getattr(self, field.name) for field in fields(self))


if __name__ == '__main__':
    pass
    import os
    import config
    import wandb
    from cs336_basics.bpe import Tokenizer, train_bpe
    from cs336_basics.transformer_language_model import TransformerLanguageModel, softmax
    
    # todo 1: modify to accept hyperparameters via argparse
    # todo 2: separate hyperparameters into dataclasses

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    print(f'using {device} device')
    bpe_train_corpus_path = config.corpus_path
    vocab_size = 2048
    special_tokens = []
    tokenizer_path = config.tokenizer_path

    # train_and_save_tokenizer(bpe_train_corpus_path, vocab_size, special_tokens, tokenizer_path)

    pretrain_corpus_path = config.corpus_path
    validation_corpus_path = config.validation_corpus_path
    corpus_ids_path = config.corpus_ids_path
    validation_corpus_ids_path = config.validation_corpus_ids_path

    # tokenize_and_save_corpus_ids(tokenizer_path, pretrain_corpus_path, corpus_ids_path)
    tokenize_and_save_corpus_ids(tokenizer_path, validation_corpus_path, validation_corpus_ids_path)

    use_wandb = True
    use_consine_lr_schedule = True
    use_gradient_clipping = True
    cosine_lr_schedule_config = CosineLRScheduleConfig()
    d_model = 1024
    num_heads = 4
    d_ff = 4096
    theta = 10000.0
    context_length = 1024
    num_layers = 12
    checkpoint_load_path = ''
    max_l2_norm = 0.0
    lr = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.999)
    batch_size = 16
    validation_batch_size = 16
    steps = 20
    save_steps = 5
    validation_steps = 5
    checkpoint_save_dir = config.checkpoint_dir
    if len(checkpoint_save_dir) != 0 and checkpoint_save_dir[-1] != '/':
        checkpoint_save_dir += '/'

    model = TransformerLanguageModel(d_model, num_heads, d_ff, theta, vocab_size, context_length, num_layers, device, dtype)
    optimizer = AdamW(model.parameters(), lr, weight_decay, betas)

    if checkpoint_load_path == '':
        iteration = 1
    if checkpoint_load_path != '':
        iteration = load_checkpoint(checkpoint_load_path, model, optimizer)

    corpus_ids = np.load(corpus_ids_path + '.npy', mmap_mode='r')
    validation_corpus_ids = np.load(validation_corpus_ids_path + '.npy', mmap_mode='r')

    if use_wandb:
        wandb.login()
        run = wandb.init(
            project='cs336-assignment1',
            config={
                'learning rate': lr,
                'steps': steps,
            },
        )

    while iteration <= steps:
        if save_steps != 0 and iteration % save_steps == 0:
            checkpoint_save_path = checkpoint_save_dir + f'iteration-{iteration}'
            save_checkpoint(model, optimizer, iteration, checkpoint_save_path)

        if use_consine_lr_schedule:
            cosine_lr_schedule_config.it = iteration
            optimizer.param_groups[0]['lr'] = cosine_lr_schedule(*cosine_lr_schedule_config.get_config())
        optimizer.zero_grad()

        input_ids, label_ids = get_batch(corpus_ids, batch_size, context_length, device)
        lm_head_output = model.forward(input_ids)
        loss = cross_entropy(softmax(lm_head_output, dim=-1), label_ids)
        print(f'step {iteration}, loss = {loss.cpu().item():.5f}')

        if use_gradient_clipping and max_l2_norm > 0:
            clip_gradient(model.parameters(), max_l2_norm)
        loss.backward()
        optimizer.step()

        if use_wandb:
            wandb.log(
                {
                    'train loss': loss,
                    'learning rate': optimizer.param_groups[0]['lr']
                }, 
                step=iteration
            )

        if validation_steps != 0 and iteration % validation_steps == 0:
            model.eval()
            validation_input_ids, validation_label_ids = get_batch(validation_corpus_ids, validation_batch_size, context_length, device) 
            # todo: modify to perform validation on the entire validation corpus
            with torch.no_grad():
                lm_head_output = model.forward(validation_input_ids)
                loss = cross_entropy(softmax(lm_head_output, dim=-1), validation_label_ids)
                print(f'step {iteration}, validation loss = {loss.cpu().item():.5f}')
            model.train()

            if use_wandb:
                wandb.log(
                    {
                        'validation loss': loss
                    }, 
                    step=iteration
                )

        iteration += 1
    
    wandb.finish()

    iteration -= 1
    checkpoint_save_path = checkpoint_save_dir + f'iteration-{iteration}'
    save_checkpoint(model, optimizer, iteration, checkpoint_save_path)