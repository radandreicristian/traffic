from pathlib import Path

import hydra
import torch.cuda
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from torch_geometric.datasets import MetrLa

from src.data.datamodule import MetrLaDataModule
from src.model.gnn import GNN
from src.model.optimizers import get_optimizer


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


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

    if model.ode_block.n_reg > 0:
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        reg_coeffs = model.reg_coeffs

        reg_loss = sum(reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs) if coeff != 0)
        loss += reg_loss

    model.reset_n_func_eval()
    loss.backward()
    optimizer.step()
    model.reset_n_func_eval()

    return loss.item()


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

    return loss, rmse


@hydra.main(config_path='conf',
            config_name="config.yaml")
def main(opt: DictConfig):
    opt = {**opt}
    n_previous_steps = opt.get('n_previous_steps')
    n_future_steps = opt.get('n_future_steps')

    data_folder = Path(__file__).parent.parent / 'data'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MetrLa(root=str(data_folder.absolute()),
                     n_previous_steps=n_previous_steps,
                     n_future_steps=n_future_steps)
    model = GNN(opt, dataset, device).to(device)

    dataloader = MetrLaDataModule(opt,
                                  dataset=dataset)

    train_dataloader = dataloader.train_dataloader()
    valid_dataloader = dataloader.val_dataloader()
    test_dataloader = dataloader.test_dataloader()

    print_model_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt, params)

    for epoch in range(opt['n_epochs']):
        for idx, batch in enumerate(train_dataloader):
            batch = [e.to(device) for e in batch]
            loss = train_step(model=model,
                              optimizer=optimizer,
                              batch=batch)
            print(f'Epoch {epoch}, Train Batch {idx}, Loss', loss)
        for idx, batch in enumerate(valid_dataloader):
            batch = [e.to(device) for e in batch]
            loss = eval_step(model=model,
                             batch=batch)

            print(f'Epoch {epoch}, Valid Batch {idx}, Loss', loss)

    for idx, batch in enumerate(test_dataloader):
        batch = [e.to(device) for e in batch]
        loss = eval_step(model=model,
                         batch=batch)
        print(f'Test Batch {idx}, Loss', loss)


if __name__ == '__main__':
    main()
