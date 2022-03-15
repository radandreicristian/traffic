"""
Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py

Refactoring with better modularity and type hinting.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.nn import MessagePassing

from model.ode.blocks import BaseOdeBlock
from model.ode.blocks.funcs.reg_funcs import create_reg_funcs

from model.residual_layer import ResidualLinear


class BaseGNN(MessagePassing, ABC):
    """An abstract base class for the graph-neural-diffusion based GNNs"""

    def __init__(self,
                 opt: dict):
        super(BaseGNN, self).__init__()

        self.return_previous_losses = False

        self.ode_block: Optional[BaseOdeBlock] = None
        self.opt = opt
        self.T = opt['time']
        self.p_dropout = opt['p_dropout_model']

        if opt['use_beltrami']:
            raise ValueError("Beltrami diffusion not implemented yet")
        else:
            self.fc_in = nn.Linear(in_features=1,
                                   out_features=opt['d_hidden'])

        self.d_hidden = opt['d_hidden']  # hidden_dim

        if self.opt['use_mlp_in']:
            self.mlp_in = nn.ModuleList([ResidualLinear(d_hidden=self.d_hidden, p_dropout=self.p_dropout),
                                         ResidualLinear(d_hidden=self.d_hidden, p_dropout=self.p_dropout)])

        if self.opt['use_mlp_out']:
            self.mlp_out = nn.ModuleList([ResidualLinear(d_hidden=self.d_hidden, p_dropout=self.p_dropout),
                                          ResidualLinear(d_hidden=self.d_hidden, p_dropout=self.p_dropout)])

        self.regressor = nn.Linear(self.d_hidden, 1)  # This is a regression task

        if self.opt['use_batch_norm']:
            self.bn_in = torch.nn.BatchNorm1d(self.d_hidden)
            # self.bn_out = torch.nn.BatchNorm2d(self.n_previous_steps)

        self.reg_funcs, self.reg_coeffs = create_reg_funcs(opt=self.opt)

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                pos_embedding: torch.Tensor = None) -> torch.Tensor:
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

    def rearrange_batch_norm(self,
                             tensor: torch.Tensor) -> torch.Tensor:
        """
        Rearrange the tensor such that batch norm can be applied, apply it and return the result.

        :param tensor: A tensor of shape (batch, nodes, hid)
        :return: The batch-normalised tensor with the same shape.
        """
        transposed_tensor = rearrange(tensor, 'batch nodes hid -> batch hid nodes')
        batch_norm_transposed_tensor = self.bn_in(transposed_tensor)
        return rearrange(batch_norm_transposed_tensor, 'batch hid nodes -> batch nodes hid')

    def augment_up(self,
                   tensor: torch.Tensor) -> torch.Tensor:
        """
        Add extra zero features to the feature tensor for ODE stability.

        :param tensor: A tensor to be taken through a neural ODE.
        :return: The input tensor, but the features dimension is doubled and filled with zeros.
        """
        zeros = torch.zeros(tensor.shape).to(self.device)
        # h (batch_size, n_nodes, d_hidden * 2)
        return torch.cat([tensor, zeros], dim=2)

    @staticmethod
    def augment_down(tensor):
        """
        Remove the extra features for ODE stability from the tensor.

        :param tensor: A tensor that was taken through a neural ODE.
        :return: The input tensor, but with features halved.
        """
        return torch.split(tensor, tensor.shape[2] // 2, dim=2)[0]

    def __repr__(self):
        return self.__class__.__name__
