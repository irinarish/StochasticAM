import torch
from torch import nn

class LpNorm(nn.Module):
    '''Lp-Normalization
        Normalizes inputs by dividing by norm(inputs, p)
    '''
    def __init__(self, p=2, eps=1e-9):
        super(LpNorm, self).__init__()
        self.eps = eps
        self.p = p

    def forward(self, inputs):
        norm = inputs.norm(self.p, -1, keepdim=True).add(self.eps)
        return inputs.div(norm)

    def extra_repr(self):
        return '{p}'.format(**self.__dict__)

class BatchCenter(nn.Module):
    '''Mini-batch centering
    computes the component-wise average across minibatch and substracts it
    '''
    def __init__(self, num_features, track_running_stats=True, use_running_stats=False):
        super(BatchCenter, self).__init__()
        self.num_features = num_features

        self.track_running_stats = track_running_stats
        self.use_running_stats = use_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('num_batches_tracked', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.num_batches_tracked.zero_()

    def forward(self, inputs):
        mean = inputs.mean(0)
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            exp_avg_factor = 1/self.num_batches_tracked.float()
            self.running_mean.mul_(1 - exp_avg_factor)
            self.running_mean.add_(exp_avg_factor*mean)
            if not self.use_running_stats:
                return inputs - mean
        return  inputs - self.running_mean

    def extra_repr(self):
        return '{num_features}, track_running_stats={track_running_stats}'.format(**self.__dict__)

class Flatten(nn.Module):
    r"""Reshapes the input tensor as a 2d tensor, where the size of the first (batch) dimension is preserved.

    Inputs: input
        - **input** (batch, num_dim1, num_dim1,...): tensor containing input features

    Outputs: output
        - **output'**: (batch, num_dim1*num_dim2*...): tensor containing the output
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
