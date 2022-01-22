import torch

from gnn_base import BaseGNN


class GNN(BaseGNN):

    def __init__(self,
                 opt: dict,
                 dataset,
                 device: torch.device):
        super(GNN, self).__init__(opt=opt,
                                  dataset=dataset,
                                  device=device)
        self.f = set_function(opt)
