import gc
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Iterable, AnyStr

import hydra
import numpy as np
import pytorch_lightning as pl
import torch.cuda
from clearml import Task, TaskTypes, Logger
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.datasets import MetrLa, MetrLaInMemory

from data.metrla_datamodule import MetrLaDataModule
from model.gdr_net import GraphDiffusionRecurrentNet
from model.lgdr_net import LatentGraphDiffusionRecurrentNet
from model.ode_net import OdeNet
from util.earlystopping import EarlyStopping

indices = {k: k // 5 - 1 for k in [5, 15, 30, 60]}


class Experiment:

    def __init__(self,
                 opt: DictConfig):
        self.opt = {**opt}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_previous_steps = self.opt.get('n_previous_steps')
        self.n_future_steps = self.opt.get('n_future_steps')

        self.dataset: Optional[Dataset] = None
        self.datamodule: Optional[pl.LightningDataModule] = None

        self.model: Optional[torch.nn.Module] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.train_dataloader: Optional[DataLoader] = None
        self.valid_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None

        self.best_model: Optional[torch.nn.Module] = None
        self.best_model_info: dict = {}
        self.best_val_rmse: float = float("inf")

        self.task: Optional[Task] = None
        self.clearml_logger: Optional[Logger] = None

        self.logger = logging.getLogger('traffic')
        self.logger.setLevel(logging.DEBUG)

        self.model_path = 'best_model.pt'

    @staticmethod
    def pretty_rmses(rmses: Dict[int, float]) -> str:
        """
        Formats RMSE scores in a more readable way.

        :param rmses: A dictionary where keys are integers representing minute offsets and the items are float values
        representing the RMSE at that time offset.
        :return: A formatted string containing the RMSEs.
        """
        return " ".join([f"{k} min.: {v:.1f} " for k, v in rmses.items()])

    def log_rmses(self,
                  title: str,
                  values: Dict[int, float],
                  iteration: int) -> None:
        """
        Logs a dictionary of RMSEs to ClearML.

        :param title:
        :param values:
        :param iteration:
        :return:
        """
        for minutes, value in values.items():
            series = f'RMSE @ {minutes} min.'
            self.clearml_logger.report_scalar(title=title,
                                              value=value,
                                              series=series,
                                              iteration=iteration)

    def print_model_params(self) -> None:
        """
        Prints the trainable parameters of the model.

        :param model:
        :return:
        """

        self.logger.debug(str(self.model))
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.debug(f"Total trainable params: {total_params}")

    @staticmethod
    def mean_rmses(rmses: List[Dict]) -> Dict:
        """
        Computes the mean of RMSEs over a batch.

        :param rmses: A list of dictionaries containing RMSEs for given time steps over a batch.
        :return: A dictionary containing mean RMSEs for given time steps over a batch.
        """
        keys = rmses[0].keys()
        mean_rmses = dict(zip(keys, [0] * len(keys)))
        for key in keys:
            mean_rmses[key] = np.mean([item[key] for item in rmses])
        return mean_rmses

    def common_step(self,
                    batch: List[torch.Tensor],
                    pos_encoding=None,
                    use_best_model: bool = False) -> Tuple[torch.tensor, dict]:
        """
        A common step for training and validation.

        :param batch: A tensor containing a batch of input data.
        :param pos_encoding: Optional positional encodings (for Beltrami).
        :return: A tuple containing the loss and the RMSEs.
        """
        x, y = batch

        # (batch_size, n_future, n_nodes)
        y = torch.squeeze(y)
        x_squeezed = torch.squeeze(x)

        if not self.model.return_previous_losses:
            y_hat = self.model(x) if not use_best_model else self.best_model(x)

            loss = mse_loss(y, y_hat)

            rmses = {k: torch.sqrt(
                torch.mean((y[:, v, :] - y_hat.detach()[:, v, :]) ** 2)).item() for k, v in indices.items()}

            del x, y, y_hat
            return loss, rmses
        else:
            y_hat_prev, y_hat_future = self.model(x) if not use_best_model else self.best_model(x)

            prev_loss = mse_loss(y_hat_prev, x_squeezed[:, 1:, :])
            future_loss = mse_loss(y_hat_future, y)
            rmses = {k: torch.sqrt(
                torch.mean((y[:, v, :] - y_hat_future.detach()[:, v, :]) ** 2)).item() for k, v in indices.items()}

            loss = prev_loss + future_loss
            del x, y, y_hat_prev, y_hat_future

            return loss, rmses

    def train_step(self,
                   batch: List[torch.Tensor],
                   pos_encoding=None) -> Tuple[torch.tensor, dict]:
        """
        A single training step, including forward, loss and backward pass for a batch.

        :param batch: A tensor containing a batch of input data.
        :param pos_encoding: Optional positional encodings (for Beltrami).
        :return: A tuple containing the loss and the RMSEs.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, rmses = self.common_step(batch, pos_encoding)

        if self.model.ode_block.n_reg > 0:
            reg_states = tuple(torch.mean(rs) for rs in self.model.reg_states)
            reg_coeffs = self.model.reg_coeffs

            reg_loss = sum(reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs) if coeff != 0)
            loss += reg_loss

        self.model.reset_n_func_eval()
        loss.backward()
        self.optimizer.step()
        self.model.reset_n_func_eval()
        loss_value = loss.item()

        del batch, loss
        torch.cuda.empty_cache()
        return loss_value, rmses

    @torch.no_grad()
    def eval_step(self,
                  batch: List[torch.Tensor],
                  pos_encoding=None,
                  use_best_model: bool = False) -> Tuple[torch.tensor, dict]:
        """
        A single evaluation (valid/test) step, including forward, loss and backward pass for a batch.

        :param batch: A tensor containing a batch of input data.
        :param pos_encoding: Optional positional encodings (for Beltrami).
        :param use_best_model: Whether to use the best model (for test set) or not.
        :return: A tuple containing the loss and the RMSEs.
        """
        if use_best_model:
            self.best_model.eval()
        else:
            self.model.eval()
        loss, rmses = self.common_step(batch, pos_encoding, use_best_model)

        return loss.item(), rmses

    def setup(self):
        """
        Setup the experiment.

        Initializes the dataset, datamodule, experiment versioning,
        :return:
        """
        inmemory_data = self.opt.get('in_memory')
        data_root_arg = self.opt.get('data_path')

        data_root = Path(__file__).parent.parent if not data_root_arg else Path(data_root_arg)

        if inmemory_data:
            data_path = data_root / 'data' / 'inmemory'
            self.dataset = MetrLaInMemory(root=str(data_path.absolute()),
                                          n_previous_steps=self.n_previous_steps,
                                          n_future_steps=self.n_future_steps)
        else:
            data_path = data_root / 'data' / 'disk'
            self.dataset = MetrLa(root=str(data_path.absolute()),
                                  n_previous_steps=self.n_previous_steps,
                                  n_future_steps=self.n_future_steps)

        self.datamodule = MetrLaDataModule(self.opt,
                                           dataset=self.dataset)

        self.train_dataloader = self.datamodule.train_dataloader()
        self.valid_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.set_model()

        self.print_model_params()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.set_optimizer(params)

        self.task = Task.init(project_name='Graph Diffusion Traffic Forecasting',
                              task_name='Train Task - V.0.1',
                              task_type=TaskTypes.training)

        self.clearml_logger = self.task.logger

    def set_model(self) -> None:
        model_tag = self.opt['model_type']

        if model_tag == 'lgdr':
            model = LatentGraphDiffusionRecurrentNet(self.opt, self.dataset, self.device).to(self.device)
        elif model_tag == 'gdr':
            model = GraphDiffusionRecurrentNet(self.opt, self.dataset, self.device).to(self.device)
        elif model_tag == 'ode':
            model = OdeNet(self.opt, self.dataset, self.device).to(self.device)
        else:
            raise ValueError('Invalid model name')
        self.model = model

    def set_optimizer(self,
                      params: Union[Iterable[torch.Tensor], Dict[AnyStr, torch.Tensor]]) -> None:
        """
        Constructs an optimizer as specified by the arguments

        :param opt: A configuration dictionary.
        :param params: Iterable or dictionary of tensors to be optimized.
        :return: A PyTorch optimizer.
        """
        optimizer_name = self.opt['optimizer']
        lr = self.opt['lr']
        weight_decay = self.opt['weight_decay']

        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params=params,
                                        lr=lr,
                                        weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(params=params,
                                            lr=lr,
                                            weight_decay=weight_decay)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(params=params,
                                            lr=lr,
                                            weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params=params,
                                         lr=lr,
                                         weight_decay=weight_decay)
        elif optimizer_name == 'adamax':
            optimizer = torch.optim.Adamax(params=params,
                                           lr=lr,
                                           weight_decay=weight_decay)
        else:
            raise ValueError("Unsupported optimizer: {}".format(optimizer_name))

        self.optimizer = optimizer

    def train(self):
        early_stopping = EarlyStopping(checkpoint_path=self.model_path)
        for epoch in range(self.opt['n_epochs']):
            train_rmses = []
            train_losses = []
            for idx, batch in enumerate(self.train_dataloader):
                start_time = time.time()
                batch = [e.to(self.device) for e in batch]
                loss, rmses = self.train_step(batch=batch)

                train_losses.append(loss)
                train_rmses.append(rmses)
                end_time = time.time()

                if idx % 50 == 0:
                    self.logger.debug(f"[Train|Ep.{epoch}|B.{idx}|{(end_time - start_time):.1f}s]: Loss {loss:.2f}, "
                                      f"RMSEs: {self.pretty_rmses(rmses)}")
                del loss, rmses

            train_loss = float(np.mean(train_losses))
            train_rmses = self.mean_rmses(train_rmses)

            self.clearml_logger.report_scalar('Losses', value=train_loss, series='Train Loss', iteration=epoch)
            self.log_rmses('Train RMSEs', values=train_rmses, iteration=epoch)

            self.logger.debug(f'[Train|Ep.{epoch}|Overall]: Loss {train_loss}, RMSEs: {self.pretty_rmses(train_rmses)}')

            valid_rmses = []
            valid_losses = []
            for idx, batch in enumerate(self.valid_dataloader):
                batch = [e.to(self.device) for e in batch]
                loss, rmses = self.eval_step(batch=batch)

                valid_losses.append(loss)
                valid_rmses.append(rmses)

            valid_loss = float(np.mean(valid_losses))
            valid_rmses = self.mean_rmses(valid_rmses)

            self.clearml_logger.report_scalar('Losses', value=valid_loss, series='Validation Loss', iteration=epoch)
            self.log_rmses('Validation RMSEs', values=valid_rmses, iteration=epoch)
            self.logger.debug(f'[Valid|Ep.{epoch}|Overall]: Loss {valid_loss}, RMSEs {self.pretty_rmses(valid_rmses)}')

            mean_valid_rmse = np.mean(list(valid_rmses.values()))
            if mean_valid_rmse < self.best_val_rmse:
                self.best_model = self.model
                self.best_val_rmse = mean_valid_rmse
                self.best_model_info = {'Epoch': epoch}
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.logger.debug(f"Early stopping triggered after epoch {epoch}. No valid loss improvement in the "
                                  f"last {early_stopping.patience} epochs.")
                self.task.update_output_model(self.model_path, "Model")
                break
        self.logger.debug(f"Best epoch: {self.best_model_info['Epoch']}. Mean RMSE: {self.best_val_rmse}")
        if not early_stopping.early_stop:
            torch.save(self.best_model.state_dict(), self.model_path)
            self.task.update_output_model(self.model_path, "Model")

    def test(self):
        test_rmses = []
        for idx, batch in enumerate(self.test_dataloader):
            batch = [e.to(self.device) for e in batch]
            _, rmses = self.eval_step(batch=batch, use_best_model=True)
            test_rmses.append(rmses)

        self.test_rmses = self.mean_rmses(test_rmses)
        self.log_rmses('Test RMSEs', values=self.test_rmses, iteration=1)
        self.logger.debug(f'[Test]: RMSEs {self.pretty_rmses(self.test_rmses)}')

    def run(self):
        self.logger.debug("Setting up the experiment.")
        self.setup()

        self.logger.debug("Starting training the model.")
        self.train()

        self.logger.debug("Starting testing the model.")
        self.test()

    def get_test_rmse(self):
        test_rmse_values = self.test_rmses.values()
        mean_test_rmse = sum(test_rmse_values) / len(test_rmse_values)
        return mean_test_rmse


@hydra.main(config_path='conf',
            config_name="config.yaml")
def main(config: DictConfig):
    experiment = Experiment(config)
    experiment.run()
    return experiment.get_test_rmse()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        gc.collect()
