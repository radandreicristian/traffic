import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Iterable, AnyStr

from adn import ADN
from clearml.logger import StdStreamPatch
import numpy as np
import pytorch_lightning as pl
import torch.cuda
from clearml import Task, TaskTypes, Logger
from einops import repeat
from omegaconf import DictConfig
from torch.nn.functional import l1_loss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.datasets import MetrLaInMemory, PemsBayInMemory

from src.util.utils import get_number_of_nodes_autoregressive
from src.data.traffic_datamodule_adn import TrafficDataModule

from src.util.constants import *
from src.util.earlystopping import EarlyStopping

import os
import binascii
from src.util.masked_metrics import masked_mae, masked_mape, masked_rmse, masked_mae_loss


indices = {k: k // 5 - 1 for k in [5, 15, 30, 60]}


class AutoregressiveExperiment:
    def __init__(self, opt: DictConfig):
        self.opt = {**opt}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_previous_steps = self.opt.get("n_previous_steps")
        self.n_future_steps = self.opt.get("n_future_steps")

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

        self.logger = logging.getLogger("traffic")

        self.model_path = "best_model.pt"

        self.positional_embeddings: Optional[torch.Tensor] = None

        self.batch_log_frequency = opt.get("batch_log_frequency", 50)
        self.batch_size = opt.get("batch_size")
        self.test_rmses: Optional[Dict] = None

        self.run_from_checkpoint = opt.get("from_checkpoint", False)
        self.task_checkpoint = opt.get("clearml_task_id")
        self.spatial_range: Optional[torch.Tensor] = None
        self.scheduler = None

    @staticmethod
    def pretty_rmses(rmses: Dict[int, float]) -> str:
        """
        Formats RMSE scores in a more readable way.

        :param rmses: A dictionary where keys are integers representing minute offsets and the items are float values
        representing the RMSE at that time offset.
        :return: A formatted string containing the RMSEs.
        """
        return " ".join([f"{k} min.: {v:.1f} " for k, v in rmses.items()])

    def log_metric(
        self, metric: str, stage: str, values: Dict[int, float], iteration: int
    ) -> None:
        """
        Logs a dictionary of RMSEs to ClearML.

        :param title:
        :param values:
        :param iteration:
        :return:
        """
        title = f"{stage} {metric}s"
        for minutes, value in values.items():
            series = f"{metric} @ {minutes} min."
            self.clearml_logger.report_scalar(
                title=title, value=value, series=series, iteration=iteration
            )

    def print_model_params(self) -> None:
        """
        Prints the trainable parameters of the model.

        :param model:
        :return:
        """

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f"Total trainable params: {total_params}")

    @staticmethod
    def mean_metric(values: List[Dict]) -> Dict:
        """
        Computes the mean of a metric over a batch.

        :param values: A list of dictionaries containing RMSEs for given time steps over a batch.
        :return: A dictionary containing mean RMSEs for given time steps over a batch.
        """
        keys = values[0].keys()
        mean_metric_value = dict(zip(keys, [0] * len(keys)))
        for key in keys:
            mean_metric_value[key] = np.mean([item[key] for item in values])
        return mean_metric_value

    @staticmethod
    def compute_metrics(y, y_hat):
        rmses = {
            k: masked_rmse(y[:, v, :], y_hat.detach()[:, v, :])
            for k, v in indices.items()
        }

        maes = {
            k: masked_mae(y[:, v, :], y_hat.detach()[:, v, :])
            for k, v in indices.items()
        }

        mapes = {
            k: masked_mape(y[:, v, :], y_hat.detach()[:, v, :])
            for k, v in indices.items()
        }

        metrics = {"rmses": rmses, "maes": maes, "mapes": mapes}
        return metrics

    def train_step(
        self, batch: List[Dict[str, torch.Tensor]], use_best_model: bool = False
    ) -> Tuple[torch.tensor, dict]:
        """
        A single training step, including forward, loss and backward pass for a batch.

        :param batch: A tensor containing a batch of input data.
        :param use_best_model: Whether to use the best model so far or not (for testing).
        :return: A tuple containing the loss and the metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        src, tgt = batch
        src_features = src["features"]

        # Feed [P11, F0, ..., F10] to predict [F0, ..., F11]
        tgt_features = torch.cat((src["features"][..., -1, :].unsqueeze(-2),
                                  tgt["features"][..., :-1, :]), dim=-2)

        model_args = {
            "src_features": src_features,
            "src_interval_of_day": src["interval_of_day"],
            "src_day_of_week": src["day_of_week"],
            "src_spatial_descriptor": self.spatial_range,
            "tgt_features": tgt_features,
            "tgt_interval_of_day": tgt["interval_of_day"],
            "tgt_day_of_week": tgt["day_of_week"],
            "tgt_spatial_descriptor": self.spatial_range
        }

        if use_best_model:
            y_hat = self.best_model(**model_args)
        else:
            y_hat = self.model(**model_args)

        # De-normalize
        y_hat = y_hat * self.train_std + self.train_mean

        loss = masked_mae_loss(tgt["raw_features"], y_hat)
        metrics = self.compute_metrics(tgt["raw_features"], y_hat)

        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()

        del batch, loss
        torch.cuda.empty_cache()
        return loss_value, metrics

    @torch.no_grad()
    def eval_step(
        self, batch: List[Dict[str, torch.Tensor]], use_best_model: bool = False
    ) -> Tuple[torch.tensor, dict]:
        """
        A single evaluation (valid/test) step, including forward, loss and backward pass for a batch.

        :param batch: A tensor containing a batch of input data.
        :param use_best_model: Whether to use the best model (for test set) or not.
        :return: A tuple containing the loss and the RMSEs.
        """
        src, tgt = batch
        src_features = src["features"]

        batch_size, _, _, _ = src_features.size()
        # Autoregressive mode - The initial tensor is
        y_hat = torch.zeros((batch_size, self.opt["n_nodes"],
                             self.n_future_steps, 1)).to(self.device)

        y_hat[..., 0, :] = src_features[..., -1, :]
        model_args = {
            "src_features": src_features,
            "src_interval_of_day": src["interval_of_day"],
            "src_day_of_week": src["day_of_week"],
            "src_spatial_descriptor": self.spatial_range,
            "tgt_features": y_hat,
            "tgt_interval_of_day": tgt["interval_of_day"],
            "tgt_day_of_week": tgt["day_of_week"],
            "tgt_spatial_descriptor": self.spatial_range
        }

        # At the end of this loop y_hat will store P(n-1), F(0), ...F(n-1)
        for i in range(1, self.n_future_steps):
            if use_best_model:
                y_hat_intermediary = self.best_model(**model_args)
            else:
                y_hat_intermediary = self.model(**model_args)
            y_hat[..., i, :] = y_hat_intermediary[..., i, :]
            model_args["tgt_features"] = y_hat

        # Forward it one more time (in the autoencoder fashion) to get F(0),..., F(n)
        if use_best_model:
            y_hat = self.best_model(**model_args)
        else:
            y_hat = self.model(**model_args)

        y_hat = y_hat * self.train_std + self.train_mean

        loss = masked_mae_loss(tgt["raw_features"], y_hat)
        metrics = self.compute_metrics(tgt["raw_features"], y_hat)

        loss_value = loss.item()

        del batch, loss
        torch.cuda.empty_cache()
        return loss_value, metrics

    def setup_data(self):
        datasets = {
            METR_LA_DATASET_NAME: MetrLaInMemory,
            PEMS_BAY_DATASET_NAME: PemsBayInMemory,
        }

        dataset_name = self.opt.get("dataset")
        data_root_arg = self.opt.get("data_path")

        data_root = (
            Path(__file__).parent.parent if not data_root_arg else Path(data_root_arg)
        )

        # This works on Path objects.
        data_path = data_root / "mem" / dataset_name

        self.dataset = datasets[dataset_name](
            root=str(data_path.absolute()),
            n_previous_steps=self.n_previous_steps,
            n_future_steps=self.n_future_steps,
        )

        self.datamodule = TrafficDataModule(self.opt, dataset=self.dataset)

        self.train_dataloader = self.datamodule.train_dataloader()
        self.valid_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.opt["n_nodes"] = get_number_of_nodes_autoregressive(self.dataset)
        spatial_range = torch.arange(start=0, end=self.opt["n_nodes"]).to(self.device)
        self.spatial_range = repeat(spatial_range, 'n -> n t', t=self.n_future_steps)

        self.train_mean, self.train_std = self.datamodule.get_normalizers()
        # Todo - This is assuming N_FUTURE = N_PAST

    def setup_new(self):
        """
        Setup the experiment.

        Initializes the dataset, datamodule, experiment versioning,
        :return:
        """
        tag = binascii.b2a_hex(os.urandom(3)).decode("utf-8")
        self.task = Task.init(
            project_name="Traffic Forecasting - ADN",
            task_name=f"Experiment {tag}",
            task_type=TaskTypes.training,
            reuse_last_task_id=False,
            output_uri="s3://traffic-models",
        )
        self.clearml_logger = self.task.logger

        self.setup_data()

        self.set_model()

        self.print_model_params()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.set_optimizer(params)

    def setup_existing(self) -> None:
        pass

    def set_model(self) -> None:
        models = {
            "adn_base": ADN,
        }

        model_tag = self.opt["model_type"]
        try:
            model_type = models[model_tag]
        except KeyError:
            raise
        self.model = model_type(
                d_features=self.opt["d_features"],
                d_hidden=self.opt["d_hidden"],
                d_feedforward=self.opt["d_feedforward"],
                n_heads=self.opt["n_heads"],
                p_dropout=self.opt["p_dropout"],
                n_blocks=self.opt["n_blocks"],
                spatial_seq_len=self.opt["n_nodes"],
                temporal_seq_len=self.n_future_steps).to(self.device)

        # Todo - Maybe remove the gradients clipping?
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, max=0.1))

    def set_optimizer(
        self, params: Union[Iterable[torch.Tensor], Dict[AnyStr, torch.Tensor]]
    ) -> None:
        """
        Constructs an optimizer as specified by the arguments

        :param opt: A configuration dictionary.
        :param params: Iterable or dictionary of tensors to be optimized.
        :return: A PyTorch optimizer.
        """
        optimizer_name = self.opt["optimizer"]
        lr = self.opt["lr"]

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(params=params, lr=lr)
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(params=params, lr=lr)
        elif optimizer_name == "adagrad":
            optimizer = torch.optim.Adagrad(params=params, lr=lr)
        elif optimizer_name == "adam":
            betas = (self.opt["beta_1"], self.opt["beta_2"])
            eps = self.opt["eps"]
            optimizer = torch.optim.Adam(
                params=params, lr=lr, betas=betas, eps=eps
            )
        elif optimizer_name == "adamax":
            optimizer = torch.optim.Adamax(params=params, lr=lr)
        else:
            raise ValueError("Unsupported optimizer: {}".format(optimizer_name))

        self.optimizer = optimizer
        self.scheduler = MultiStepLR(self.optimizer, milestones=[15, 30, 45], gamma=0.5)

    def train(self):
        patience = self.opt["early_stop_patience"]
        early_stopping = EarlyStopping(
            checkpoint_path=self.model_path, patience=patience
        )
        for epoch in range(self.opt["n_epochs"]):
            train_rmses = []
            train_maes = []
            train_mapes = []
            train_losses = []
            for idx, batch in enumerate(self.train_dataloader):
                start_time = time.time()
                batch = [{k: v.to(self.device) for k, v in e.items()} for e in batch]
                loss, metrics = self.train_step(batch=batch)

                train_losses.append(loss)
                train_rmses.append(metrics["rmses"])
                train_maes.append(metrics["maes"])
                train_mapes.append(metrics["mapes"])

                end_time = time.time()

                if idx % self.batch_log_frequency == 0:
                    self.logger.info(
                        f"[Train|Ep.{epoch}|B.{idx}|{(end_time - start_time):.1f}s]: "
                        f"Loss {loss:.2f}"
                    )
                del loss, metrics

            train_loss = float(np.mean(train_losses))
            train_rmses = self.mean_metric(train_rmses)
            train_maes = self.mean_metric(train_maes)
            train_mapes = self.mean_metric(train_mapes)

            self.clearml_logger.report_scalar(
                "Losses", value=train_loss, series="Train Loss", iteration=epoch
            )
            self.log_metric("RMSE", "Train", values=train_rmses, iteration=epoch)
            self.log_metric("MAE", "Train", values=train_maes, iteration=epoch)
            self.log_metric("MAPE", "Train", values=train_mapes, iteration=epoch)

            self.logger.info(f"[Train|Ep.{epoch}|Overall]: Loss {train_loss:.2f}.")

            valid_rmses = []
            valid_maes = []
            valid_mapes = []
            valid_losses = []

            for idx, batch in enumerate(self.valid_dataloader):
                batch = [{k: v.to(self.device) for k, v in e.items()} for e in batch]
                loss, metrics = self.eval_step(batch=batch)

                valid_losses.append(loss)
                valid_rmses.append(metrics["rmses"])
                valid_maes.append(metrics["maes"])
                valid_mapes.append(metrics["mapes"])

            valid_loss = float(np.mean(valid_losses))
            valid_rmses = self.mean_metric(valid_rmses)
            valid_maes = self.mean_metric(valid_maes)
            valid_mapes = self.mean_metric(valid_mapes)

            self.clearml_logger.report_scalar(
                "Losses", value=valid_loss, series="Validation Loss", iteration=epoch
            )
            self.log_metric("RMSE", "Validation", values=valid_rmses, iteration=epoch)
            self.log_metric("MAE", "Validation", values=valid_maes, iteration=epoch)
            self.log_metric("MAPE", "Validation", values=valid_mapes, iteration=epoch)

            self.logger.info(f"[Valid|Ep.{epoch}|Overall]: Loss {valid_loss:.2f}.")

            mean_valid_rmse = np.mean(list(valid_rmses.values()))
            if mean_valid_rmse < self.best_val_rmse:
                self.best_model = self.model
                self.best_val_rmse = mean_valid_rmse
                self.best_model_info = {"Epoch": epoch}
                torch.save(self.best_model.state_dict(), self.model_path)
                self.task.update_output_model(self.model_path)

            self.scheduler.step()
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info(
                    f"Early stopping triggered after epoch {epoch}. No valid loss improvement in the "
                    f"last {early_stopping.patience} epochs."
                )
                self.task.update_output_model(self.model_path)
                break

        self.logger.info(
            f"Best epoch: {self.best_model_info['Epoch']}. Mean RMSE: {self.best_val_rmse}"
        )
        if not early_stopping.early_stop:
            torch.save(self.best_model.state_dict(), self.model_path)
            # self.task.update_output_model(self.model_path, "Model")
        self.task.update_output_model(self.model_path)

    def test(self):
        test_rmses = []
        test_maes = []
        test_mapes = []
        for idx, batch in enumerate(self.test_dataloader):
            batch = [{k: v.to(self.device) for k, v in e.items()} for e in batch]
            _, metrics = self.eval_step(batch=batch, use_best_model=True)
            test_rmses.append(metrics["rmses"])
            test_maes.append(metrics["maes"])
            test_mapes.append(metrics["mapes"])

        test_rmses = self.mean_metric(test_rmses)
        test_maes = self.mean_metric(test_maes)
        test_mapes = self.mean_metric(test_mapes)

        self.log_metric("RMSE", "Test", values=test_rmses, iteration=1)
        self.log_metric("MAE", "Test", values=test_maes, iteration=1)
        self.log_metric("MAPE", "Test", values=test_mapes, iteration=1)
        self.logger.info("Evaluating on test data complete.")

        # this is for optuna
        self.test_rmses = test_rmses

    def _resume_previous_experiment(self):
        self.setup_existing()

        self.logger.info("Resuming training the model.")

    def _run_new_experiment(self):

        self.setup_new()

        self.logger.info("Starting training the model.")

        try:
            self.train()
        except KeyboardInterrupt:
            print("Force stopped training. Evaluating on test set.")
        finally:
            self.logger.info("Starting testing the model.")
            self.test()

        self.task.close()

    def run(self):
        StdStreamPatch.remove_std_logger()

        if self.run_from_checkpoint:
            self.logger.info("Resuming previous experiment.")
            self._resume_previous_experiment()
        else:
            self.logger.info("Setting up the experiment.")
            self._run_new_experiment()

    def get_test_rmse(self):
        test_rmse_values = self.test_rmses.values()
        mean_test_rmse = sum(test_rmse_values) / len(test_rmse_values)
        return mean_test_rmse
