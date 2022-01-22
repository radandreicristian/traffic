"""
Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py

Refactoring with better modularity and type hinting.
"""
from typing import List, Callable, Union, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.types
from torch import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.loop import add_remaining_self_loops

from src.ode.utils import squareplus


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
                 data,
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
        self.attention_weights = None

        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))

        self.x0 = None
        self.n_func_eval = 0

        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

    def __repr__(self):
        return self.__class__.__name__


class RegularizedOdeFunc(nn.Module):
    """
    A regularized OdeFunc
    """

    def __init__(self,
                 ode_func: BaseOdeFunc,
                 reg_funcs: List[Callable]) -> None:
        """
        Initializes the Regularized ODE object.

        :param ode_func: An ODE function.
        :param reg_funcs: A list of callable regularization functions.
        """
        super(RegularizedOdeFunc, self).__init__()

        self.ode_func = ode_func
        self.reg_funcs = reg_funcs

    def forward(self,
                t: torch.Tensor,
                state: torch.Tensor) -> Union[torch.Tensor, Tuple[Any]]:
        """
        Forwards the

        :param t:
        :param state:
        :return:
        """
        with torch.enable_grad():
            x = state[0]
            x.requires_grad_(True)
            t.requires_grad_(True)
            d_state = self.ode_func(t, x)
            if len(state) > 1:
                dx = d_state
                reg_states = tuple(reg_func(x, t, dx, self.ode_func) for reg_func in self.reg_funcs)
                return (d_state,) + reg_states
            else:
                return d_state

    @property
    def _num_evals(self):
        return self.ode_func._num_evals


class TransformerAttentionOdeFunc(BaseOdeFunc):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 opt: dict,
                 data: torch.utils.data.Dataset,
                 device: torch.device):
        super(TransformerAttentionOdeFunc, self).__init__(in_features=in_features,
                                                          out_features=out_features,
                                                          opt=opt,
                                                          data=data,
                                                          device=device)
        if opt['self_loop_weight'] > 0:
            self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index)


class SparseGraphTransformerAttention(nn.Module):
    """
    From https://github.com/twitter-research/graph-neural-pde/blob/main/src/function_transformer_attention.py
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 opt: dict,
                 device: torch.device,
                 concat: bool = True,
                 edge_weights: torch.Tensor = None):
        super(SparseGraphTransformerAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.opt = opt
        self.device = device
        self.concat = concat
        self.edge_weights = edge_weights

        self.n_heads = int(opt['heads'])
        self.alpha = opt['leaky_relu_slope']

        try:
            self.d_attention = opt['attention_dim']
        except KeyError:
            self.d_attention = out_features

        assert self.d_attention % self.n_heads == 0, f"Number of heads {self.h} must be a factor of d {self.d_attention}"
        self.d_head = self.d_attention // self.n_heads

        # todo: add beltrami options

        self.q = nn.Linear(in_features=in_features,
                           out_features=self.d_attention)

        self.v = nn.Linear(in_features=in_features,
                           out_features=self.d_attention)

        self.k = nn.Linear(in_features=in_features,
                           out_features=self.d_attention)

        self.init_weights(self.q)
        self.init_weights(self.k)
        self.init_weights(self.v)

    def forward(self,
                x: torch.Tensor,
                edge) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # todo: add beltrami opts

        q = self.q(x).view(-1, self.h, self.d_head)
        k = self.k(x).view(-1, self.h, self.d_head)
        v = self.v(x).view(-1, self.h, self.d_head)

        src = q[edge[0, :], :, :]
        dst_k = k[edge[1, :], :, :]

        if self.opt['attention_type'] == "scaled_dot":
            prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_head)
        else:
            raise NotImplementedError("Refer the GRAND paper src code.")

        if self.opt['reweight_attention'] and self.edge_weights:
            prods *= self.edge_weights.unsqueeze(dim=1)

        if self.opt['square_plus']:
            attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
        else:
            attention = softmax(prods, edge[self.opt['attention_norm_idx']])

        return attention, (v, prods)

    def init_weights(self,
                     layer: nn.Module):
        nn.init.constant_(layer.weight, 1e-5)


class LaplacianOdeFunc(BaseOdeFunc):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 opt: dict,
                 data,
                 device: torch.device) -> None:
        super(LaplacianOdeFunc, self).__init__(opt=opt,
                                               data=data,
                                               device=device)

        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
