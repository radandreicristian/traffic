from typing import Type

from model.ode.blocks.attention import AttentionOdeBlock
from model.ode.blocks.base import BaseOdeBlock


def get_ode_block(opt: dict) -> Type[BaseOdeBlock]:
    function = opt.get('function', "")
    if function == 'transformer':
        block = AttentionOdeBlock
    else:
        raise ValueError(f"Block type {function} not defined.")
    return block
