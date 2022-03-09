"""
Regularized ODE function module.

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/regularized_ODE_function.py
"""
from typing import List, Callable, Union, Tuple, Any

import torch
import torch.nn as nn

from model.ode.blocks.funcs.base import BaseOdeFunc


class RegularizedOdeFunc(nn.Module):
    """
    A regularized OdeFunc.

    :param ode_func: An ODE function.
    :param reg_funcs: A list of callable regularization functions.
    """

    def __init__(self,
                 ode_func: BaseOdeFunc,
                 reg_funcs: List[Callable]) -> None:
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
        return self.ode_func.n_func_eval
