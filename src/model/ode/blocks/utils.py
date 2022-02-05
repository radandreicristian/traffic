"""
Utilities for graph ODEs.

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/utils.py
"""

import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


def get_rw_adj(edge_index,
               edge_attr=None,
               norm_dim=1,
               fill_value: float = 0.,
               num_nodes=None,
               dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_attr is None:
        edge_attr = torch.ones((edge_index.size(1),), dtype=dtype,
                               device=edge_index.device)

    fill_value = float(fill_value)
    if not fill_value == 0.:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index=edge_index,
            edge_attr=edge_attr,
            fill_value=fill_value,
            num_nodes=num_nodes)
        assert tmp_edge_weight is not None
        edge_attr = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    deg = scatter_add(edge_attr, indices, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    edge_attr = deg_inv_sqrt[indices] * edge_attr if norm_dim == 0 else edge_attr * deg_inv_sqrt[indices]
    return edge_index, edge_attr
