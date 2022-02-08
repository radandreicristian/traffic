"""
Scaled dot product ODE function.

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/function_transformer_attention.py
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.types
import torch_sparse
from torch_geometric.utils import softmax
from torch_geometric.utils.loop import add_remaining_self_loops

from src.model.ode.blocks.funcs.base import BaseOdeFunc
from src.model.ode.utils import squareplus


class ScaledDotProductOdeFunc(BaseOdeFunc):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 device: torch.device,
                 opt: dict) -> None:
        super(ScaledDotProductOdeFunc, self).__init__(in_features=in_features,
                                                      out_features=out_features,
                                                      opt=opt,
                                                      edge_index=edge_index,
                                                      edge_attr=edge_attr,
                                                      device=device)
        if opt['self_loop_weight'] > 0:
            self.edge_index, self.edge_attr = add_remaining_self_loops(edge_index,
                                                                       edge_attr,
                                                                       fill_value=opt[
                                                                           'self_loop_weight'])

        else:
            self.edge_index, self.edge_attr = edge_index, edge_attr

        # self.batch_size = opt['batch_size']
        self.in_features = in_features

        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        # self.edge_index = self.edge_index.to(self.device)
        # self.edge_attr = self.edge_attr.to(self.device)
        self.attention_layer = SparseGraphTransformerAttention(in_features=in_features,
                                                               out_features=out_features,
                                                               opt=opt,
                                                               edge_attr=self.edge_attr,
                                                               device=self.device)

    def apply_attention(self, x, attention, v=None):

        if self.opt['mix_features']:
            vx = torch.mean(torch.stack(
                [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
                 range(self.opt['n_heads'])], dim=0),
                dim=0)
            ax = self.attention_layer.w_out(vx)
        else:
            mean_attention = attention.mean(dim=1)
            ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
        return ax

    def forward(self, t, x):
        if self.n_func_eval > self.opt['max_func_evals']:
            pass

        self.n_func_eval += 1
        attention, (values, _) = self.attention_layer(x, self.edge_index)
        ax = self.apply_attention(x, attention, values)

        ax = torch.reshape(ax, [-1, 207, self.in_features])

        if self.opt['alpha_sigmoid']:
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train
        f = alpha * (ax - x)
        if self.opt['add_source']:
            f = f + self.beta_train * self.x0
        return f


class SparseGraphTransformerAttention(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 opt: dict,
                 device,
                 concat: bool = True,
                 edge_attr: torch.Tensor = None,
                 ) -> None:
        super(SparseGraphTransformerAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.opt = opt
        self.concat = concat
        self.edge_attr = edge_attr.cuda()

        self.n_heads = int(opt['n_heads'])
        self.alpha = opt['leaky_relu_slope']

        self.device = device
        try:
            self.d_attention = opt['d_hidden_attention']
        except KeyError:
            self.d_attention = out_features

        assert self.d_attention % self.n_heads == 0, f"N heads {self.n_heads} must be a factor of d {self.d_attention}"

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

        self.activation = nn.Sigmoid()
        self.w_out = nn.Linear(self.d_head, in_features)

    def forward(self,
                x: torch.Tensor,
                edge) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # if "cuda" not in str(edge.device):
        # print("well, this shouldn't happen...")
        # print("moving to...", self.device)
        # edge = edge.to(self.device)

        q = self.q(x).view(-1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(-1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(-1, self.n_heads, self.d_head).transpose(1, 2)

        src = q[edge[0, :], :, :]
        dst_k = k[edge[1, :], :, :]

        if self.opt['attention_type'] == "scaled_dot":
            prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_head)
        else:
            raise NotImplementedError("""Rest of methods are unimplemented for the moment. This is the best option for
            the initial experiments. For different attentions, see GRAND src code:
            /src/function_attention_transformer.py)""")

        if self.opt['reweight_attention'] and self.edge_attr is not None:
            prods *= self.edge_attr.unsqueeze(dim=1)

        if self.opt['square_plus']:
            attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
        else:
            attention = softmax(prods, edge[self.opt['attention_norm_idx']])

        return attention, (v, prods)

    @staticmethod
    def init_weights(module: nn.Module):
        if type(module) == nn.Linear:
            nn.init.constant_(module.weight, 1e-5)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(
            self.out_features) + ')'
