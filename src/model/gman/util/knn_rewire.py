from typing import Optional, Union, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def knn_rewire(
    edge_index: Tensor,
    k: int,
    edge_attr: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    pass

    adj_dense = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr).squeeze()
    n_nodes = adj_dense.shape[0]

    # Select the indices of the top k values on each row
    _, indices = torch.topk(adj_dense, k, dim=1, sorted=False)

    topk_adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for node in range(n_nodes):
        top_k = indices[node, :]
        for neighbour in top_k:
            topk_adj[node, neighbour] = adj_dense[node, neighbour]

    edge_index, edge_attr = dense_to_sparse(topk_adj)
    return edge_index, edge_attr
