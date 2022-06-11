import logging

import torch
import numpy as np
from torch.nn.functional import l1_loss, mse_loss

logger = logging.getLogger('metrics')


def make_mask(y_true,
              null_value):
    # If null value is Nan
    if np.isnan(null_value):
        # Mask = {False, if y_true == NaN, True if y_true != NaN
        mask = ~torch.isnan(y_true)
    else:
        # Mask = {False, if y_true == null_value, True if y_true != null_value
        mask = torch.not_equal(y_true, null_value)
    # Convert to 0. and 1.
    mask = mask.to(torch.float32)

    mask /= torch.mean(mask)

    # Put zero where the mask is nan
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mae(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               null_value=0) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    mae = torch.abs(y_true - y_pred) * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)

    return torch.mean(mae).item()


def masked_rmse(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                null_value=0) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    rmse = (y_true - y_pred) ** 2 * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    return torch.sqrt(torch.mean(rmse)).item()


def masked_mape(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                null_value=0) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    mape = (torch.abs(y_true - y_pred) / y_true) * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    return torch.mean(mape).item()


def masked_mae_loss(y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    null_value=0) -> torch.Tensor:

    mask = make_mask(y_true=y_true, null_value=null_value)

    mae = torch.abs(y_true - y_pred) * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)

    return torch.mean(mae)


if __name__ == '__main__':
    y_true = torch.Tensor([[1, 0, 3, 2],
                           [1, 2, 0, 2]])

    y_pred = torch.Tensor([[0.9, 1, 2.5, 2],
                           [0.5, 2, 2, 2]])

    # mae = 0.1+1+0.5+0+0.5+0+2+0=4.1/8=0.5125
    # mae_mask = 0.1+0+0.5+0+0.5+0+0+0=0.18(3)
    mae = l1_loss(y_true, y_pred)
    mae_mask = masked_mae(y_true, y_pred)
    print(mae)
    print(mae_mask)

    # rmse = 0.01 + 1 + 0.25 + 0 + 0.25 + 0 + 4 + 0 = sqrt(5.51/8)= 0.829
    # rmse_mask = 0.01 + 0 + 0.25 + 0 + 0.25 + 0 + 0 + 0 = sqrt(0.51/6) = 0.2915
    rmse = torch.sqrt(mse_loss(y_true, y_pred))
    rmse_mask = masked_rmse(y_true, y_pred)
    print(rmse)
    print(rmse_mask)

    # mape = 0.1/1 + 1/0 + 0.5/3 + 0 + 0.5/1 + 0 + 2/0 + 0 = 0.1+0+0.16+0.5=0.76/6=0.13
    mape = masked_mape(y_true, y_pred)
    print(mape)
