from ode.ode_blocks import AttentionOdeBlock


def set_block(opt):
    ode_str = opt['block']
    if ode_str == 'attention':
        block = AttentionOdeBlock
    else:
        raise ValueError(f"Block type {ode_str} not defined.")
    return block


def set_function(opt):
    ode_str = opt['function']
    if ode_str == 'laplacian':
        f = LaplacianODEFunc
    elif ode_str == 'GAT':
        f = ODEFuncAtt
    elif ode_str == 'transformer':
        f = ODEFuncTransformerAtt
    else:
        raise FunctionNotDefined
    return f
