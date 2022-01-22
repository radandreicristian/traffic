from typing import Optional

import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, gather_csr, segment_csr, scatter


def get_rw_adj(edge_index,
               edge_weight=None,
               norm_dim=1,
               fill_value: float = 0.,
               num_nodes=None,
               dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    fill_value = float(fill_value)
    if not fill_value == 0.:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=fill_value,
            num_nodes=num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
    return edge_index, edge_weight


def squareplus(src: torch.Tensor,
               index: Optional[torch.Tensor],
               ptr: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
    r"""Computes a sparsely evaluated softmax.
      Given a value tensor :attr:`src`, this function first groups the values
      along the first dimension based on the indices specified in :attr:`index`,
      and then proceeds to compute the softmax individually for each group.

      Args:
          src (Tensor): The source tensor.
          index (LongTensor): The indices of elements for applying the softmax.
          ptr (LongTensor, optional): If given, computes the softmax based on
              sorted inputs in CSR representation. (default: :obj:`None`)
          num_nodes (int, optional): The number of nodes, *i.e.*
              :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

      :rtype: :class:`Tensor`
      """
    out = src - src.max()
    # out = out.exp()
    out = (out + torch.sqrt(out ** 2 + 4)) / 2

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)
