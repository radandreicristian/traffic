import logging

import torch
from pykeops.torch import LazyTensor

logger = logging.getLogger("rewiring")


@torch.no_grad()
def knn_rewire(
    model,
    features,
    opt,
):
    k = opt["rewire_KNN_k"]
    x_i = LazyTensor(features[:, None, :])
    x_j = LazyTensor(features[None, :, :])

    distances = ((x_i - x_j) ** 2).sum(-1)
    indices = distances.argKmin(k, dim=-1)

    linspace = (
        torch.linspace(
            start=0,
            end=len(indices.view(-1)),
            steps=len(indices.view(-1)) + 1,
            dtype=torch.int64,
            device=indices.device,
        )[:-1].unsqueeze(0)
        // k
    )
