import torch


def make_mask(y_true,
              null_value):
    mask = ~torch.isnan(y_true) if null_value else torch.not_equal(y_true, null_value)
    mask = mask.to(torch.float32)
    mask /= mask.mean()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mae(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               null_value=torch.nan) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    mae = torch.abs(y_true - y_pred) * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    return torch.mean(mae).item()


def masked_rmse(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                null_value=torch.nan) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    rmse = (y_true - y_pred) ** 2 * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    return torch.sqrt(torch.mean(rmse)).item()


def masked_mape(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                null_value=torch.nan) -> torch.Tensor:
    mask = make_mask(y_true=y_true, null_value=null_value)

    mape = torch.abs(y_true - y_pred) / (y_true + 1e-7) * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    return torch.mean(mape).item()


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