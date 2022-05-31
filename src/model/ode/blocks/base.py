"""
Base class for ODE layers.

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/base_classes.py
"""
from typing import List, Callable, Optional
from typing import Type

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from src.model.ode.blocks.funcs.base import BaseOdeFunc
from src.model.ode.blocks.funcs.regularized import RegularizedOdeFunc


class BaseOdeBlock(nn.Module):
    """A layer that consists of a differentiable ODE solver."""

    def __init__(
        self,
        ode_func: Type[BaseOdeFunc],
        reg_funcs: List[Callable],
        opt: dict,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_nodes: int,
        device,
        t: torch.Tensor,
    ) -> None:
        """
        Initializes the ODE Block Layer.

        :param ode_func: A type of a subclass of OdeFunc (uninstantiated).
        :param reg_funcs: A list of regularization functions.
        :param opt: A dictionary containing command line options.
        :param t: A tensor containing the evaluation points (in time). First element should be 0.

        :return: None.
        """
        super(BaseOdeBlock, self).__init__()
        self.device = device

        self.opt: dict = opt
        self.t: torch.Tensor = t
        self.aug_dim: int = 2 if opt["use_augmentation"] else 1
        self.d_hidden = opt["d_hidden"]
        self.n_nodes = n_nodes
        self.in_features = self.out_features = self.aug_dim * self.d_hidden
        self.ode_func: BaseOdeFunc = ode_func(
            in_features=self.in_features,
            out_features=self.out_features,
            opt=opt,
            edge_index=edge_index,
            edge_attr=edge_attr,
            device=device,
        )

        self.reg_ode_func = RegularizedOdeFunc(
            ode_func=self.ode_func, reg_funcs=reg_funcs
        )
        self.n_reg = len(reg_funcs)
        # The train integrator is the actual neural ODE solver.
        self.train_integrator = odeint_adjoint if opt["adjoint"] else odeint
        self.test_integrator = None

        self.default_rtol = 1e-7
        self.default_atol = 1e-9

        self.atol: Optional[float] = None
        self.rtol: Optional[float] = None
        self.adjoint_atol: Optional[float] = None
        self.adjoint_rtol: Optional[float] = None
        self.tol_scale_factor: float = self.opt["tol_scale"]

        self.set_tol()

    def set_x0(self, x0: torch.Tensor) -> None:
        """
        Sets t

        :param x0:
        :type x0:
        :return:
        :rtype:
        """
        self.ode_func.x0 = x0.clone().detach()
        self.reg_ode_func.ode_func.x0 = x0.clone().detach()

    def set_tol(self) -> None:
        """
        Set the absolute/relative tolerance rates according to the scaling
        factor and their default values.

        :return: None
        """
        self.atol = self.tol_scale_factor * self.default_atol
        self.rtol = self.tol_scale_factor * self.default_rtol
        if self.opt["adjoint"]:
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

    def set_time(self, time: int) -> None:
        """
        Set the current evaluation point (in time) in the time tensor.

        :param time The evaluation point (in time).
        :type time: int.

        :return: None.
        :rtype: NoneType.
        """
        self.t = torch.tensor([0, time])

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(Time Interval {str(self.t[0].item())} -> {str(self.t[1].item())})"
        )
