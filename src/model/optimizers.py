from collections import Iterable
from typing import Dict, Union, AnyStr

import torch


def get_optimizer(opt: dict,
                  params: Union[Iterable[torch.Tensor], Dict[AnyStr, torch.Tensor]]) -> torch.optim.Optimizer:
    """
    Constructs an optimizer as specified by the arguments

    :param opt: A configuration dictionary.
    :param params: Iterable or dictionary of tensors to be optimized.
    :param lr: Learning rate of the optimizer.
    :param weight_decay: Weight decay of the optimizer.
    :return: A PyTorch optimizer.
    """
    optimizer_name = opt['optimizer']
    lr = opt['lr']
    weight_decay = opt['weight_decay']

    if optimizer_name == 'sgd':
        return torch.optim.SGD(params=params,
                               lr=lr,
                               weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(params=params,
                                   lr=lr,
                                   weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad(params=params,
                                   lr=lr,
                                   weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(params=params,
                                lr=lr,
                                weight_decay=weight_decay)
    elif optimizer_name == 'adamax':
        return torch.optim.Adamax(params=params,
                                  lr=lr,
                                  weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_name))
