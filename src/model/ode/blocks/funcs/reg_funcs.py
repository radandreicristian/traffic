"""
Regularization functions for ODEs.

Adapted/improved/copied from GRAND/BLEND Code Repository (Apache License)
https://github.com/twitter-research/graph-neural-pde/blob/main/src/regularized_ODE_function.py
"""
from typing import Tuple, List, Callable

import torch.autograd


def create_reg_funcs(opt: dict) -> Tuple[List[Callable], List[float]]:
    """
    Create regularization functions as specified by the provided options.

    :param opt: A dictionary of options.
    :type opt: dict
    :return: A list of functions and a list of their coefficients.
    :rtype: Tuple[List[Callable], List[float]]
    """
    reg_funcs = []
    reg_coeffs = []

    for name, reg_func in all_reg_funcs.items():
        if opt[name]:
            reg_funcs.append(reg_func)
            reg_coeffs.append(opt[name])
    return reg_funcs, reg_coeffs


def directional_derivative_reg_func(
    x: torch.Tensor, t: torch.Tensor, dx: torch.Tensor, unused_context
) -> torch.Tensor:
    """
    Todo.

    :param x: An array containing the values of a function at specific time indices.
    :param t: The time indices.
    :param dx: The derivative of the function at the specific time indices.
    :return: The regularization term.
    """
    del t

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    return 0.5 * directional_dx.pow(2).view(x.size(0), -1).mean(dim=-1)


def total_derivative_reg_func(
    x: torch.Tensor, t: torch.Tensor, dx: torch.Tensor, unused_context
) -> torch.Tensor:
    """
    Todo.

    :param x: An array containing the values of a function at specific time indices.
    :param t: The time indices.
    :param dx: The derivative of the function at the specific time indices.
    :return: The regularization term.
    """
    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1 / x.numel(), requires_grad=True)
        tmp = torch.autograd.grad((u * dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        derivative = directional_dx + partial_dt

        return 0.5 * derivative.pow(2).view(x.size(0), -1).mean(dim=-1)
    except RuntimeError as e:
        if "One of the differentiated Tensors" in e.__str__():
            raise RuntimeError(
                'No partial derivative wrt time. Use "directional_derivative" regularizer instead'
            )


def jacobian_frobeniuns_reg_func(
    x: torch.Tensor, t: torch.Tensor, dx: torch.Tensor, unused_context
) -> torch.Tensor:
    """
    Regularization of the Jacobian with its Frobenius norm: https://arxiv.org/pdf/2002.02798.pdf

    :param x: An array containing the values of a function at specific time indices.
    :param t: The time indices.
    :param dx: The derivative of the function at the specific time indices.
    :return: The regularization term.
    """
    del t
    sum_diag = torch.tensor([0.0])
    for j in range(x.shape[0]):
        x_ = x[j, :, :]
        dx_ = dx[j, :, :]
        for i in range(x_.shape[1]):
            sum_diag += (
                torch.autograd.grad(
                    dx_[:, i].sum(), x_, create_graph=True, allow_unused=True
                )[0]
                .contiguous()[:, i]
                .contiguous()
            )
    return sum_diag.contiguous()


def kinetic_energy_reg_func(
    x: torch.Tensor, t: torch.Tensor, dx: torch.Tensor, unused_context
) -> torch.Tensor:
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
    "kinetic_energy": kinetic_energy_reg_func,
}

if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # t = torch.tensor([0., 1.], requires_grad=True)
    dx = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    directional_derivative = torch.autograd.grad(
        dx, x, dx, create_graph=True, allow_unused=True
    )
    print(directional_derivative)
