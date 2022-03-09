"""
Transformer attention ODE block:

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/block_transformer_attention.py
"""

from typing import Type, List, Callable, Union, Tuple, Any

import torch
from torchdiffeq import odeint_adjoint, odeint

from model.ode.blocks.base import BaseOdeBlock
from model.ode.blocks.funcs.base import BaseOdeFunc
from model.ode.blocks.funcs.scaled_dot_attention import SparseGraphTransformerAttention
from model.ode.blocks.utils import get_rw_adj


class AttentionOdeBlock(BaseOdeBlock):
    def __init__(self,
                 ode_func: Type[BaseOdeFunc],
                 reg_funcs: List[Callable],
                 opt: dict,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 n_nodes: int,
                 device,
                 t=torch.tensor([0, 1]),
                 gamma=.5):
        super(AttentionOdeBlock, self).__init__(ode_func=ode_func,
                                                reg_funcs=reg_funcs,
                                                opt=opt,
                                                edge_index=edge_index,
                                                edge_attr=edge_attr,
                                                n_nodes=n_nodes,
                                                t=t,
                                                device=device)

        self.ode_func = ode_func(in_features=self.in_features,
                                 out_features=self.out_features,
                                 opt=opt,
                                 edge_index=edge_index,
                                 edge_attr=edge_attr,
                                 device=device)

        edge_index, edge_attr = get_rw_adj(edge_index,
                                           edge_attr=edge_attr,
                                           norm_dim=1,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=n_nodes,
                                           dtype=torch.float32)

        self.ode_func.edge_index = edge_index.to(device)
        self.ode_func.edge_attr = edge_attr.to(device)

        ode_integrator = odeint_adjoint if opt['adjoint'] else odeint
        self.train_integrator = ode_integrator
        self.test_integrator = ode_integrator

        self.set_tol()
        self.attention_layer = SparseGraphTransformerAttention(in_features=self.in_features,
                                                               out_features=self.out_features,
                                                               opt=opt,
                                                               edge_attr=self.ode_func.edge_attr,
                                                               device=device)

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
                method=self.opt['solver'],
                options=dict(step_size=self.opt["step_size"], max_iters=self.opt['max_iters']),
                adjoint_method=self.opt["adjoint_solver"],
                adjoint_options=dict(step_size=self.opt["step_size"], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.adjoint_atol,
                adjoint_rtol=self.adjoint_rtol)
        else:
            state_dt = integrator(
                func, state, t,
                method=self.opt['solver'],
                options=dict(step_size=self.opt["step_size"], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol)

        if self.training and self.n_reg > 0:
            z = state_dt[0][1]
            reg_states = tuple(state[1] for state in state_dt[1:])
            return z, reg_states
        else:
            z = state_dt[1]
            return z
