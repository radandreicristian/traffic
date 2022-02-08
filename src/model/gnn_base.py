"""
Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py

Refactoring with better modularity and type hinting.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from src.model.ode.blocks import BaseOdeBlock
from src.model.ode.blocks.funcs.reg_funcs import create_reg_funcs


class BaseGNN(MessagePassing, ABC):
    """An abstract base class for the graph-neural-diffusion based GNNs"""

    def __init__(self,
                 opt: dict,
                 n_features: int,
                 n_nodes: int):
        super(BaseGNN, self).__init__()

        self.ode_block: Optional[BaseOdeBlock] = None
        self.opt = opt
        self.T = opt['time']

        # Todo - This is a regression task, so...
        # self.n_classes = dataset.num_classes
        self.n_features = n_features
        self.n_nodes = n_nodes

        if opt['use_beltrami']:
            self.fc_in_feat = nn.Linear(in_features=self.n_features,
                                        out_features=opt['d_hidden_feat'])  # feat_hidden_dim
            self.fc_in_pos = nn.Linear(in_features=opt['d_pos_enc'],  # pos_enc_dim
                                       out_features=opt['d_hidden_pos_enc'])  # pos_enc_hidden_dim
            opt['d_hidden'] = opt['d_hidden_feat'] + opt['d_hidden_pos_enc']  # hidden_dim
        else:
            self.fc_in = nn.Linear(in_features=self.n_features,
                                   out_features=opt['d_hidden'])

        self.d_hidden = opt['d_hidden']  # hidden_dim

        if self.opt['use_mlp_in']:
            self.mlp_in = nn.ModuleList([nn.Linear(in_features=self.d_hidden,
                                                   out_features=self.d_hidden),
                                         nn.Linear(in_features=self.d_hidden,
                                                   out_features=self.d_hidden)])

        if self.opt['use_mlp_out']:
            self.mlp_out = nn.ModuleList([nn.Linear(in_features=self.d_hidden,
                                                    out_features=self.d_hidden)])

        self.regressor = nn.Linear(self.d_hidden, self.opt['n_future_steps'])  # This is a regression task
        # Todo - Do I need an activation at the end?

        if self.opt['use_batch_norm']:
            self.bn_in = torch.nn.BatchNorm1d(self.d_hidden)
            # self.bn_out = torch.nn.BatchNorm2d(self.n_previous_steps)

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

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__
