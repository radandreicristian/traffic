from typing import Type

from src.model.ode.blocks.funcs.base import BaseOdeFunc
from src.model.ode.blocks.funcs.scaled_dot_attention import ScaledDotProductOdeFunc


def get_ode_function(opt: dict) -> Type[BaseOdeFunc]:
    function = opt.get('function', "")
    if function == 'transformer':
        block = ScaledDotProductOdeFunc
    else:
        raise ValueError(f"Block type {function} not defined.")
    return block
