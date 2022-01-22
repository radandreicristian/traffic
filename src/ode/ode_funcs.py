"""
Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py

Refactoring with better modularity and type hinting.
"""
import torch.nn as nn
import torch.types
from torch_geometric.nn.conv import MessagePassing


class BaseOdeFunc(MessagePassing):
    """
    Base class for ODE function layers.
    These layers should be thought as message passing, and their update rule should contain a differentiable,
    permutation-invariant function iterating over a node's neighbours and a differentiable function (i.e. MLP) that
    composes the message using the node (and edge) features.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 opt: dict,
                 data: torch.Tensor,
                 device: torch.device):
        """
        Initializes a base ODE function.

        :param in_features: Size of the input features.
        :type in_features: int.
        :param out_features: Size of the output features.
        :type out_features: int.
        :param opt: Dictionary of options.
        :type opt: dict
        :param data: The data tensor.
        :type data: torch.Tensor
        :param device: The device where the computations are run.
        :type device: torch.types.Device
        """
        super(BaseOdeFunc, self).__init__()
        self.opt = opt
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attn_weights = None

        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))

        self.x0 = None
        self.n_func_eval = 0

        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

    def __repr__(self):
        return self.__class__.__name__
