import time
from pathlib import Path

import hydra
import numpy as np
import torch.cuda
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from torch_geometric.datasets import MetrLa, MetrLaInMemory

from src.data.datamodule import MetrLaDataModule
from src.data.inmemorydatamodule import MetrLaInMemoryDataModule
from src.model.gnn import GNN
from src.model.optimizers import get_optimizer


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total params {total_params}")


def train_step(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               batch,
               pos_encoding=None):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    y = torch.squeeze(y[:, -1, :, :])
    y_hat = model(x)

    loss = mse_loss(y, y_hat)
    rmse = torch.sqrt(torch.mean((y - y_hat) ** 2)).item()

    if model.ode_block.n_reg > 0:
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        reg_coeffs = model.reg_coeffs

        reg_loss = sum(reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs) if coeff != 0)
        loss += reg_loss

    model.reset_n_func_eval()
    loss.backward()
    optimizer.step()
    model.reset_n_func_eval()

    return loss.item(), rmse


@torch.no_grad()
def eval_step(model: torch.nn.Module,
              batch,
              pos_encoding=None):
    model.eval()
    x, y = batch
    y = torch.squeeze(y[:, -1, :, :])

    y_hat = model(x)
    loss = mse_loss(y, y_hat)
    rmse = torch.sqrt(torch.mean((y - y_hat) ** 2)).item()

    return loss.item(), rmse


@hydra.main(config_path='conf',
            config_name="config.yaml")
def main(opt: DictConfig):
    opt = {**opt}
    n_previous_steps = opt.get('n_previous_steps')
    n_future_steps = opt.get('n_future_steps')

    inmemory_data = opt.get('in_memory')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inmemory_data:
        data_folder = Path(__file__).parent.parent / 'data' / 'inmemory'
        dataset = MetrLaInMemory(root=str(data_folder.absolute()),
                                 n_previous_steps=n_previous_steps,
                                 n_future_steps=n_future_steps)
        datamodule = MetrLaInMemoryDataModule(opt,
                                              dataset=dataset)

    else:
        data_folder = Path(__file__).parent.parent / 'data' / 'disk'
        dataset = MetrLa(root=str(data_folder.absolute()),
                         n_previous_steps=n_previous_steps,
                         n_future_steps=n_future_steps)
        datamodule = MetrLaDataModule(opt,
                                      dataset=dataset)

    model = GNN(opt, dataset, device).to(device)

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    print_model_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt, params)

    best_model = None
    best_model_info = None
    best_val_rmse = float("inf")

    for epoch in range(opt['n_epochs']):
        for idx, batch in enumerate(train_dataloader):
            start = time.time()
            batch = [e.to(device) for e in batch]
            loss, rmse = train_step(model=model,
                                    optimizer=optimizer,
                                    batch=batch)
            end = time.time()
            print(f'[Train]: Epoch {epoch}, Batch {idx}, Loss {loss}, RMSE {rmse}, Time {int(end - start)}s')

        val_rmses = []
        for idx, batch in enumerate(valid_dataloader):
            start = time.time()
            batch = [e.to(device) for e in batch]
            loss, rmse = eval_step(model=model,
                                   batch=batch)
            end = time.time()
            val_rmses.append(rmse)
            print(f'[Valid]: Epoch {epoch}, Batch {idx}, Loss {loss}, RMSE {rmse}, Time {int(end - start)}s')
        val_rmse = sum(val_rmses) / len(val_rmses)

        print(f'[Valid]: Epoch {epoch}, Valid RMSE {val_rmse}')
        if val_rmse < best_val_rmse:
            best_model = model
            best_val_rmse = val_rmse
            best_model_info = {'epoch': epoch}

    print(f"Best performing epoch {best_model_info['epoch']}")
    test_rmses = []
    for idx, batch in enumerate(test_dataloader):
        batch = [e.to(device) for e in batch]
        loss, rmse = eval_step(model=best_model,
                               batch=batch)
        test_rmses.append(rmse)
    test_rmse = sum(test_rmses) / len(test_rmses)
    print(f'[Test]: RMSE {test_rmse}')


if __name__ == '__main__':
    main()
