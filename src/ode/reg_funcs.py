"""
Regularization functions for ODEs.

Source: GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/regularized_ODE_function.py
"""

import torch.autograd


def directional_derivative_reg_func(x: torch.Tensor,
                                    t: torch.Tensor,
                                    dx: torch.Tensor) -> torch.Tensor:
    """
    Todo.

    :param x:
    :param t:
    :param dx:
    :return: The regularization term.
    """
    del t

    directional_dx = torch.autograd.grad(dx, x, dx,
                                         create_graph=True)[0]

    return .5 * directional_dx.pow(2).view(x.size(0), -1).mean(dim=-1)


def total_derivative_reg_func(x: torch.Tensor,
                              t: torch.Tensor,
                              dx: torch.Tensor) -> torch.Tensor:
    """
    Todo.

    :param x:
    :param t:
    :param dx:
    :return: The regularization term.
    """
    directional_dx = torch.autograd.grad(dx, x, dx,
                                         create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1 / x.numel(), requires_grad=True)
        tmp = torch.autograd.grad((u * dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        derivative = directional_dx + partial_dt

        return .5 * derivative.pow(2).view(x.size(0), -1).mean(dim=-1)
    except RuntimeError as e:
        if 'One of the differentiated Tensors' in e.__str__():
            raise RuntimeError('No partial derivative wrt time. Use "directional_derivative" regularizer instead')


def jacobian_frobeniuns_reg_func(x: torch.Tensor,
                                 t: torch.Tensor,
                                 dx: torch.Tensor) -> torch.Tensor:
    """
    Regularization of the Jacobian with its Frobenius norm: https://arxiv.org/pdf/2002.02798.pdf

    :param x:
    :param t:
    :param dx:
    :return: The regularization term.
    """
    del t
    sum_diag = torch.tensor([0.0])
    for i in range(x.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def kinetic_energy_reg_func(x: torch.Tensor,
                            t: torch.Tensor,
                            dx: torch.Tensor) -> torch.Tensor:
    """
    Quadratic cost / kinetic energy regularization: https://arxiv.org/pdf/2002.02798.pdf

    :param x:
    :param t:
    :param dx:
    :return: The regularization term.
    """

    del x, t
    dx = dx.view(dx.shape[0], -1)
    return 0.5 * dx.pow(2).mean(dim=-1)


all_reg_funcs = {
    "directional_derivative": directional_derivative_reg_func,
    "total_derivative": total_derivative_reg_func,
    "jacobian_norm2": jacobian_frobeniuns_reg_func,
    "kinetic_energy": kinetic_energy_reg_func
}
