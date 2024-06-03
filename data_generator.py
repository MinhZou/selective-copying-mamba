import torch
import random
import numpy as np
import torch.nn.functional as F

def torch_copying_data(L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
    """
    Generate a dataset for a sequence copying task.
    This code is adopted from the copying.py script in the S4 repository. The original code can be found at:
    https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

    Parameters:
    L (int): Number of padding tokens
    M (int): Number of tokens to memorize
    A (int): Alphabet size
    variable (bool): If True, selective copying task
    variable_length (bool): If True, randomize number of tokens to memorize
    batch_shape (tuple): Shape of the batch
    one_hot (bool): If True, convert the input sequence into a one-hot encoded tensor
    reverse (bool): If True, reverse the order of the target sequence

    Returns:
    tuple: Generated input sequence and target sequence
    """
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch)
            ], 0)
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse: y_ = y_.flip(-1)
    if one_hot: x = F.one_hot(x_, A).float()
    else: x = x_
    y = y_
    return x, y

"""
Examples:
print(torch_copying_data(10, 5, 10, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
print(torch_copying_data(10, 5, 10, variable=True, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
Outputs:
(tensor([2, 2, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([2, 2, 2, 4, 6])) # copying memory task
(tensor([0, 6, 0, 0, 0, 0, 0, 6, 7, 0, 7, 5, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([6, 6, 7, 7, 5])) # selective copying task
"""
def generate_dataset(dataset_config, training_config):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    dataset_config (dict): Configuration for the dataset
    training_config (dict): Configuration for the training

    Returns:
    tuple: Generated inputs and targets
    """
    x, y  = torch_copying_data(dataset_config["l_noise"], dataset_config["l_memorize"], dataset_config["n_tokens"],
                              batch_shape=(training_config["batch_size"],),variable=dataset_config["variable"],
                              variable_length=dataset_config["variable_length"], one_hot=dataset_config["one_hot"],
                              reverse=dataset_config["reverse"])
    return x, y


