import torch
import numpy as np


def mean_ignore_zeros(tensor):
    nonzero_elements = torch.count_nonzero(tensor)
    return torch.sum(tensor) / nonzero_elements


def make_mask(y_true,
              null_value):
    # If null value is Nan
    if np.isnan(null_value):
        # Mask = {False, if y_true == NaN, True if y_true != NaN
        mask = ~torch.isnan(y_true)
    else:
        # Mask = {False, if y_true == null_value, True if y_true != null_value
        mask = torch.not_equal(y_true, null_value)
    # Convert to 0-1
    mask = mask.to(torch.float32)

    # Put zero where the mask is nan
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mae(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               null_value=0) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    mae = torch.abs(y_true - y_pred) * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)

    return mean_ignore_zeros(mae).item()


def masked_rmse(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                null_value=0) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    rmse = (y_true - y_pred) ** 2 * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    return torch.sqrt(mean_ignore_zeros(rmse)).item()


def masked_mape(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                null_value=0) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    mape = (torch.abs(y_true - y_pred) / y_true) * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    return (mean_ignore_zeros(mape)).item()


def masked_mae_loss(y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    null_value=0) -> torch.Tensor:

    mask = make_mask(y_true=y_true, null_value=null_value)

    mae = torch.abs(y_true - y_pred) * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)

    return mean_ignore_zeros(mae)



if __name__ == '__main__':
    y_true = torch.randn((2, 315, 12, 32))
    y_pred = torch.randn((2, 315, 12, 32))

    y_true[1, 2, 3, 4] = 0
    y_true[1, 2, 4, 4] = 0

    mae = masked_mae(y_true, y_pred)
    rmse = masked_rmse(y_true, y_pred)
    mape = masked_mape(y_true, y_pred)

    print(mae)
    print(rmse)
    print(mape)