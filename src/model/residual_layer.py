import torch.nn as nn


class ResidualLinear(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 p_dropout: float):
        super(ResidualLinear, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(in_features=d_hidden,
                                out_features=d_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.dropout(x + self.linear(self.relu(x)))
