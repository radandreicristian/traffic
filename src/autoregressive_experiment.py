import binascii
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Iterable, AnyStr

import numpy as np
import pytorch_lightning as pl
import torch.cuda
import torch.nn as nn
from adn import ADN
from clearml import Task, TaskTypes, Logger
from clearml.logger import StdStreamPatch
from einops import repeat
from omegaconf import DictConfig
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.datasets import MetrLaInMemory, PemsBayInMemory

from src.data.traffic_datamodule_adn import TrafficDataModule
from src.util.constants import *
from src.util.earlystopping import EarlyStopping
from src.util.masked_metrics import masked_mae, masked_mape, masked_rmse, \
    masked_mae_loss
from src.util.utils import get_number_of_nodes_autoregressive

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
        self.best_val_metric: float = float("inf")

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
        self.normalized_loss = opt.get("normalized_loss")
        self.autoregressive_validation = opt.get("autoregressive_validation")

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
            k: masked_rmse(y[:, :, v, :], y_hat.detach()[:, :, v, :])
            for k, v in indices.items()
        }

        maes = {
            k: masked_mae(y[:, :, v, :], y_hat.detach()[:, :, v, :])
            for k, v in indices.items()
        }

        mapes = {
            k: masked_mape(y[:, :, v, :], y_hat.detach()[:, :, v, :])
            for k, v in indices.items()
        }

        metrics = {"rmses": rmses, "maes": maes, "mapes": mapes}
        return metrics

    def get_forward_kwargs(self):
        if self.attention_type == "group":
            return {"partitions": self.model.get_partitions()}
        else:
            return {}

    def autoencoder_step(self,
                         batch: List[Dict[str, torch.Tensor]]
                         ) -> Tuple[torch.tensor, dict]:

        src_dict, tgt_dict = batch

        # src (B, N, T, 1)
        src = src_dict["features"]

        # tgt (B, N, T, 1)
        tgt = tgt_dict["features"]

        # Feed [P11, F0, ..., F10] to predict [F0, ..., F11]
        # Concatenate (B, N, 1, 1) with (B, N, T-1, 1)
        tgt_inp = torch.cat((src[..., -1, :].unsqueeze(-2),
                             tgt[..., :-1, :]), dim=-2)

        model_args = {
            "src_features": src,
            "src_interval_of_day": src_dict["interval_of_day"],
            "src_day_of_week": src_dict["day_of_week"],
            "src_spatial_descriptor": self.spatial_range,
            "tgt_features": tgt_inp,
            "tgt_interval_of_day": tgt_dict["interval_of_day"],
            "tgt_day_of_week": tgt_dict["day_of_week"],
            "tgt_spatial_descriptor": self.spatial_range,
            **self.get_forward_kwargs()
        }

        tgt_out = self.model(**model_args)

        # De-normalize
        tgt_out_denormalized = tgt_out * self.train_std + self.train_mean

        if self.normalized_loss:
            loss = masked_mae_loss(tgt_dict["features"], tgt_out)
        else:
            loss = masked_mae_loss(tgt_dict["raw_features"], tgt_out_denormalized)

        # loss = l1_loss(tgt_dict["features"], tgt_out)
        # loss = l1_loss(tgt_dict["raw_features"], tgt_out_denorm)
        # loss = masked_mae_loss(tgt_dict["features"], tgt_out)
        metrics = self.compute_metrics(tgt_dict["raw_features"], tgt_out_denormalized)

        return loss, metrics

    def autoregressive_step(self,
                            batch,
                            use_best_model=False):

        model = self.best_model if use_best_model else self.model

        src_dict, tgt_dict = batch

        src = src_dict["features"]

        # Initialize the decoder input as a sequence of zeros of MAX_LEN (12)
        tgt_inp = torch.zeros_like(src)

        # Place P11 on first position in decoder input
        tgt_inp[..., 0, :] = src[..., -1, :]

        model_args = {
            "src_features": src,
            "src_interval_of_day": src_dict["interval_of_day"],
            "src_day_of_week": src_dict["day_of_week"],
            "src_spatial_descriptor": self.spatial_range,
            "tgt_features": tgt_inp,
            "tgt_interval_of_day": tgt_dict["interval_of_day"],
            "tgt_day_of_week": tgt_dict["day_of_week"],
            "tgt_spatial_descriptor": self.spatial_range,
            **self.get_forward_kwargs()
        }

        for i in range(1, 12):
            tgt_out = model(**model_args)
            tgt_inp[..., i, :] = tgt_out[..., i - 1, :]
            model_args["tgt_features"] = tgt_inp

        tgt_out = model(**model_args)

        tgt_out = torch.cat((tgt_inp[..., 1:, :], tgt_out[..., -1, :].unsqueeze(-2)),
                            dim=-2)
        tgt_out_denormalized = tgt_out * self.train_std + self.train_mean

        if self.normalized_loss:
            loss = masked_mae_loss(tgt_dict["features"], tgt_out)
        else:
            loss = masked_mae_loss(tgt_dict["raw_features"], tgt_out_denormalized)

        metrics = self.compute_metrics(tgt_dict["raw_features"], tgt_out_denormalized)

        return loss, metrics

    def train_step(self,
                   batch: List[Dict[str, torch.Tensor]]
                   ) -> Tuple[torch.tensor, dict]:
        """
        A single training (autoencoder) step.

        :param batch: A tensor containing a batch of input data.
        :return: A tuple containing the loss and the metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self.autoencoder_step(batch=batch)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

        self.optimizer.step()
        loss_value = loss.item()

        del batch, loss
        torch.cuda.empty_cache()
        return loss_value, metrics

    @torch.no_grad()
    def valid_step(self, batch: List[Dict[str, torch.Tensor]]):
        """
        A single validation (autoencoder) step.

        :param batch: A tensor containing a batch of input data.
        :return: A tuple containing the loss and the metrics.
        """
        if self.autoregressive_validation:
            loss, metrics = self.autoregressive_step(batch=batch)
        else:
            loss, metrics = self.autoencoder_step(batch=batch)

        loss_value = loss.item()

        del batch, loss
        torch.cuda.empty_cache()
        return loss_value, metrics

    @torch.no_grad()
    def test_step(self, batch: List[Dict[str, torch.Tensor]]) -> Dict:
        """
        A single autoregressive test step.

        :param batch: A tensor containing a batch of input data.
        :return: A tuple containing the loss and the RMSEs.
        """

        _, metrics = self.autoregressive_step(batch, use_best_model=True)
        torch.cuda.empty_cache()
        return metrics

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
            auto_connect_frameworks={
                "pytorch" : ['best_model.pt']
            }
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
        self.attention_type = self.opt["attention_type"]
        attention_kwargs = self.opt["attention_config"].get(self.attention_type, {})

        try:
            model_type = models[model_tag]
        except KeyError:
            raise

        model_kwargs = {
            "d_features": self.opt["d_features"],
            "d_hidden": self.opt["d_hidden"],
            "d_feedforward": self.opt["d_feedforward"],
            "n_heads": self.opt["n_heads"],
            "p_dropout": self.opt["p_dropout"],
            "n_blocks": self.opt["n_blocks"],
            "spatial_seq_len": self.opt["n_nodes"],
            "temporal_seq_len": self.n_future_steps,
            "spatial_attention_type": self.attention_type
        }
        self.model = model_type(**model_kwargs, **attention_kwargs).to(self.device)

        # Todo - Maybe remove the gradients clipping?
        # for p in self.model.parameters():
        #    p.register_hook(lambda grad: torch.clamp(grad, max=0.1))

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
            if self.attention_type == "group":
                self.model.set_partitions()
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
                loss, metrics = self.valid_step(batch=batch)

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

            mean_valid_rmse = np.mean(list(valid_maes.values()))
            if mean_valid_rmse < self.best_val_metric:
                self.best_model = self.model
                self.best_val_metric = mean_valid_rmse
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
            f"Best epoch: {self.best_model_info['Epoch']}. Mean RMSE: {self.best_val_metric}"
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
            metrics = self.test_step(batch=batch)
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
            print(f"Force stopped training. Evaluating with {self.best_model_info}.")
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
