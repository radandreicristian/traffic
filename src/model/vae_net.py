import torch.nn as nn
import torch_geometric.data
from torch.nn import GRU


class RecurrentEncoder(nn.Module):
    def __init__(self, opt: dict, dataset: torch_geometric.data.Dataset) -> None:
        super(RecurrentEncoder, self).__init__()

        self.d_hidden = opt.get("d_hidden")
        self.encoder = GRU(
            input_size=self.d_hidden,
            hidden_size=self.d_hidden,
            num_layers=1,
            batch_first=True,
        )
