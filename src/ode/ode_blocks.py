"""
Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py

Refactoring with better modularity and type hinting.
"""
from typing import Callable, List, Type, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from ode_funcs import BaseOdeFunc, RegularizedOdeFunc, SparseGraphTransformerAttention
from utils import get_rw_adj


class OdeBlock(nn.Module):
    """A layer that consists of a differentiable ODE solver."""

    def __init__(self,
                 ode_func: Type[BaseOdeFunc],
                 reg_funcs: List[Callable],
                 opt: dict,
                 data,
                 device: torch.device,
                 t: torch.Tensor) -> None:
        """
        Initializes the ODE Block Layer.

        :param ode_func: A type of a subclass of OdeFunc (uninstantiated).
        :param reg_funcs: A list of regularization functions.
        :param opt: A dictionary containing command line options.
        :param data: A tensor containing the data batch.
        :param device: The device on which the pytorch computations happen (CPU/GPU)
        :param t: A tensor containing the evaluation points (in time). First element should be 0.

        :return: None.
        """
        super(OdeBlock, self).__init__()

        self.opt: dict = opt
        self.t: torch.Tensor = t
        self.aug_dim: int = 2 if opt['augment'] else 1
        self.d_hidden = opt['hidden_dim']
        self.in_features = self.out_features = self.aug_dim * self.d_hidden
        self.ode_func: BaseOdeFunc = ode_func(in_features=self.in_features,
                                              out_features=self.out_features,
                                              opt=opt,
                                              data=data,
                                              device=device)

        self.reg_ode_func = RegularizedOdeFunc(ode_func=self.ode_func,
                                               reg_funcs=reg_funcs)
        self.n_reg = len(reg_funcs)
        # The train integrator is the actual neural ODE solver.
        self.train_integrator = odeint_adjoint if opt['adjoint'] else odeint
        self.test_integrator = None

        self.default_rtol = 1e-7
        self.default_atol = 1e-9

        self.atol: Optional[float] = None
        self.rtol: Optional[float] = None
        self.adjoint_atol: Optional[float] = None
        self.adjoint_rtol: Optional[float] = None
        self.tol_scale_factor: float = self.opt['tol_scale_factor']

        self.set_tol()

    def set_x0(self, x0: torch.Tensor) -> None:
        """
        Sets t

        :param x0:
        :type x0:
        :return:
        :rtype:
        """
        self.ode_func.x0 = x0.clone().detatch()
        self.reg_ode_func.ode_func.x0 = x0.clone().detatch()

    def set_tol(self) -> None:
        """
        Set the absolute/relative tolerance rates according to the scaling factor and their default values.

        :return: None
        """
        self.atol = self.tol_scale_factor * self.default_atol
        self.rtol = self.tol_scale_factor * self.default_rtol
        if self.opt['adjoint']:
            self.adjoint_atol = self.tol_scale_factor * self.default_atol
            self.adjoint_rtol = self.tol_scale_factor * self.default_rtol

    def reset_tol(self) -> None:
        """
        Reset the absolute/relative tolerance rates to their default values.

        :return: None.
        """
        self.atol = self.default_atol
        self.rtol = self.default_rtol
        self.adjoint_atol = self.default_atol
        self.adjoint_rtol = self.default_rtol

    def set_time(self,
                 time: int) -> None:
        """
        Set the current evaluation point (in time) in the time tensor.

        :param time The evaluation point (in time).
        :type time: int.

        :return: None.
        :rtype: NoneType.
        """
        self.t = torch.tensor([0, time]).to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + f"(Time Interval {str(self.t[0].item())} -> {str(self.t[1].item())})"


class AttentionOdeBlock(OdeBlock):
    def __init__(self,
                 ode_func: Type[BaseOdeFunc],
                 reg_funcs: List[Callable],
                 opt: dict,
                 data,
                 device: torch.device,
                 t=torch.tensor([0, 1]),
                 gamma=.5):
        super(AttentionOdeBlock, self).__init__(ode_func=ode_func,
                                                reg_funcs=reg_funcs,
                                                opt=opt,
                                                data=data,
                                                device=device,
                                                t=t)

        self.ode_func = ode_func(in_features=self.in_features,
                                 out_features=self.out_features,
                                 opt=opt,
                                 data=data,
                                 device=device)

        edge_index, edge_weight = get_rw_adj(data.edge_index,
                                             edge_weight=data.edge_attr,
                                             norm_dim=1,
                                             fill_value=opt['self_loop_weight'],
                                             num_nodes=data.num_nodes,
                                             dtype=data.x.type)

        self.ode_func.edge_index = edge_index.to(device)
        self.ode_func.edge_weight = edge_weight.to(device)

        ode_integrator = odeint_adjoint if opt['adjoint'] else odeint
        self.train_integrator = ode_integrator
        self.test_integrator = ode_integrator

        self.set_tol()
        self.attention_layer = SparseGraphTransformerAttention(in_features=self.d_hidden,
                                                               out_features=self.d_hidden,
                                                               opt=opt,
                                                               device=device,
                                                               edge_weights=self.ode_func.edge_weight).to(device)

    def get_attention_weights(self, x):
        attention, _ = self.attention_layer(x=x,
                                            edge=self.ode_func.edge_index)
        return attention

    def forward(self,
                x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[Any]]]:
        t = self.t.type_as(x)
        self.ode_func.attention_weights = self.get_attention_weights(x)
        self.reg_ode_func.ode_func.attention_weights = self.ode_func.attention_weights

        integrator = self.train_integrator if self.training else self.test_integrator

        reg_states = tuple(torch.zeros(x.size(0)).to(x) for _ in range(self.n_reg))

        func = self.reg_ode_func if self.training and self.n_reg > 0 else self.ode_func
        state = (x,) + reg_states if self.training and self.n_reg > 0 else x

        if self.opt["adjoint"] and self.training:
            state_dt = integrator(
                func, state, t,
                method=self.opt['method'],
                options={"step_size": self.opt["step_size"]},
                adjoint_method=self.opt["adjoint_method"],
                adjoint_options={"step_size": self.opt["adjoint_step_size"]},
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.adjoint_atol,
                adjoint_rtol=self.adjoint_rtol)
        else:
            state_dt = integrator(
                func, state, t,
                method=self.opt['method'],
                options={"step_size": self.opt["step_size"]},
                atol=self.atol,
                rtol=self.rtol)

        if self.training and self.n_reg > 0:
            z = state_dt[0][1]
            reg_states = tuple(state[1] for state in state_dt[1:])
            return z, reg_states
        else:
            z = state_dt[1]
            return z
