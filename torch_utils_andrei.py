import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor

def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')
    
    if random_state is None:
        random_state = np.random.RandomState()

    
    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def _repr_model(model):

    if model._net is None:
        net_representation = '[uninitialised]'
    else:
        net_representation = repr(model._net)

    return ('<{}: {}>'
            .format(
                model.__class__.__name__,
                net_representation,
            ))


def set_seed(seed, cuda=False):

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


# def minibatch(batch_size, *tensors):
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)
    
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def regression_loss(observed_ratings, predicted_ratings):
    assert_no_grad(observed_ratings)
    
    return ((observed_ratings - predicted_ratings) ** 2).mean()




def _repr_model(model):

    if model._net is None:
        net_representation = '[uninitialised]'
    else:
        net_representation = repr(model._net)

    return ('<{}: {}>'
            .format(
                model.__class__.__name__,
                net_representation,
            ))


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the emedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)



class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the emedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)