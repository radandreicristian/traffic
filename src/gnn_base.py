"""
Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py

Refactoring with better modularity and type hinting.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Callable

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from ode.ode_blocks import OdeBlock
from ode.reg_funcs import all_reg_funcs


def create_reg_funcs(opt: dict) -> Tuple[List[Callable], List[float]]:
    """
    Create regularization functions as specified by the provided options.

    :param opt: A dictionary of options.
    :type opt: dict
    :return: A list of functions and a list of their coefficients.
    :rtype: Tuple[List[Callable], List[float]]
    """
    reg_funcs = []
    reg_coeffs = []

    for name, reg_func in all_reg_funcs.items():
        if opt[name]:
            reg_funcs.append(reg_func)
            reg_coeffs.append(opt[name])
    return reg_funcs, reg_coeffs


class BaseGNN(MessagePassing, ABC):
    """An abstract base class for the graph-neural-diffusion based GNNs"""

    def __init__(self,
                 opt: dict,
                 dataset,
                 device: torch.device = torch.device("cpu")):
        super(BaseGNN, self).__init__()

        # The concrete implementations have an instantiated ode_block. This is just a base class anyway.
        self.ode_block: Optional[OdeBlock] = None

        self.opt = opt
        self.T = opt['time']

        # Todo - This is a regression task, so...
        # self.n_classes = dataset.num_classes
        self.n_features = dataset.data.n_features
        self.n_nodes = dataset.data.n_nodes

        if opt['beltrami']:
            self.mx = nn.Linear(in_features=self.n_features,
                                out_features=opt['d_hidden_feat'])  # feat_hidden_dim
            self.mp = nn.Linear(in_features=opt['d_pos_enc'],  # pos_enc_dim
                                out_features=opt['d_hidden_pos_enc'])  # pos_enc_hidden_dim
            opt['d_hidden'] = opt['d_hidden_feat'] + opt['d_hidden_pos_enc']  # hidden_dim
        else:
            self.m1 = nn.Linear(in_features=self.n_features,
                                out_features=opt['d_hidden'])

        d_hidden = opt['d_hidden']  # hidden_dim

        if self.opt['use_mlp']:
            self.m11 = nn.Linear(in_features=d_hidden,
                                 out_features=d_hidden)  # hidden_dim
            self.m12 = nn.Linear(in_features=d_hidden,
                                 out_features=d_hidden)  # hidden_dim
        if opt['fc_out']:
            self.fc = nn.Linear(in_features=d_hidden,
                                out_features=d_hidden)

        self.m2 = nn.Linear(d_hidden, 1)  # This is a regression task, so the output should be numeric.
        self.hidden_dim = opt['d_hidden']  # hidden_dim

        if self.opt['batch_norm']:
            self.bn_in = torch.nn.BatchNorm1d(d_hidden)
            self.bn_out = torch.nn.BatchNorm1d(d_hidden)

        self.reg_funcs, self.reg_coeffs = create_reg_funcs(opt=self.opt)

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                pos_embedding: torch.Tensor) -> torch.Tensor:
        pass

    def get_n_func_eval(self) -> int:
        """
        Get the current number of function evaluations.

        :return: The current number of function evaluations.
        :rtype: int.
        """
        return self.ode_block.ode_func.n_func_eval + self.ode_block.reg_ode_func.ode_func.n_func_eval

    def reset_n_func_eval(self) -> None:
        """
        Reset the ODE block's ODE func and Reg ODE func's number of evaluations.

        :return: None.
        :rtype: NoneType.
        """
        self.ode_block.ode_func.n_func_eval = 0
        self.ode_block.reg_ode_func.ode_func.n_func_eval = 0

    def __repr__(self):
        return self.__class__.__name__
