"""
Regularization functions for the Neural ODEs.

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/regularized_ODE_function.py
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class BaseOdeFunc(MessagePassing):
    """
    Base class for ODE function layers.

    These layers should be thought as message passing, and their update rule should contain a differentiable,
    permutation-invariant function iterating over a node's neighbours and a differentiable function (i.e. MLP)
    that composes the message using the node (and edge) features.

    :param in_features: Size of the input features.
    :param out_features: Size of the output features.
    :param opt: Dictionary of options.
    :param edge_index: Paris of adjacent node indices (sparse COO format).
    :param edge_attr: Adjacency weight (sparse COO format).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        opt: dict,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        device,
    ) -> None:
        super(BaseOdeFunc, self).__init__()
        self.opt = opt
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.attention_weights = None

        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))

        self.x0 = None
        self.n_func_eval = 0

        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

        self.device = device

    def __repr__(self):
        return self.__class__.__name__
