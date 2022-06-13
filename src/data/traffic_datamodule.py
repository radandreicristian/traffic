import logging
from collections import Iterable
from typing import Optional, Callable, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Data, Dataset

from src.util.constants import IN_MEMORY, ON_DISK


class TrafficDataModule(pl.LightningDataModule):
    """A data module that holds the dataloaders for the MetrLa dataset."""

    def __init__(self, opt: dict, dataset: Dataset) -> None:
        """
        Initialize the DataModule.
        :param opt: A dictionary of options.
        :param dataset: The dataset object.
        """
        super(TrafficDataModule, self).__init__()
        self.logger = logging.getLogger("traffic")

        self.opt = opt
        self.dataset = dataset
        self.batch_size = opt["batch_size"]

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        dataset_loading_location = opt.get("dataset_loading_location")

        collate_fns = {
            IN_MEMORY: self.inmemory_collate_fn,
            ON_DISK: self.ondisk_collate_fn,
        }

        self.collate_fn = collate_fns[dataset_loading_location]

        self.train_mean: Optional[torch.Tensor] = None
        self.train_std: Optional[torch.Tensor] = None
        self.setup()

    def sample(self, indices: Sequence, split: str) -> Sequence:
        """
        Return sampled indices based on the configuration.

        If the configuration specifies to sample the dataset, take the sample factor for the specified split and return
        values sampled from the initial indices.
        :param indices: A sequence (iterable, sized, reversible) of indices of data points in the original dataset.
        :param split: The name of the split (train/valid/test).
        :return: A numpy array containing either indices or a subset of it.
        """
        if self.opt["sample_dataset"]:
            sample_factor = f"sample_{split}_factor"
            factor = self.opt[sample_factor]
            rng = np.random.default_rng(21)
            indices = rng.choice(
                indices, int(float(factor) * len(indices)), replace=False
            )

        return indices

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the DataModule.

        :param stage: Unused.
        :return: None.
        """
        dataset_len = len(self.dataset)
        valid_percentage = self.opt["valid_percentage"]
        test_percentage = self.opt["test_percentage"]
        train_percentage = 1.0 - (test_percentage + valid_percentage)

        last_train_index = int(train_percentage * dataset_len)
        last_valid_index = int((train_percentage + valid_percentage) * dataset_len)

        train_indices = range(0, last_train_index)
        valid_indices = range(last_train_index, last_valid_index)
        test_indices = range(last_valid_index, dataset_len)

        train_indices = self.sample(train_indices, "train")
        valid_indices = self.sample(valid_indices, "valid")
        test_indices = self.sample(test_indices, "test")

        print(f"Samples: {len(train_indices)}/{len(valid_indices)}/{len(test_indices)}")

        self.logger.debug(
            f"Samples: {len(train_indices)}/{len(valid_indices)}/{len(test_indices)}"
        )
        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        train_features = torch.stack([x[:, :, 0] for x, _ in self.train_dataset])

        self.train_mean = torch.mean(train_features)
        self.train_std = torch.std(train_features)

    @staticmethod
    def onehot_temporal(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert temporal indices (hour of day and day of week) into one-hot encoded feature vector.

        :param tensor: A tensor of shape (*, 2), where (..., 0) is the hour-of-day and (..., 1) is the day-of-week
        feature.
        :return: A tensor of shape (*, 31), where the last dimension is the concatenation of one-hot-encoded
        hour-of-day (24d) and day-of-week (7d) vector.
        """
        hour_of_day = tensor[..., 0]
        day_of_week = tensor[..., 1]

        hour_of_day_one_hot = f.one_hot(hour_of_day.to(torch.long), num_classes=24)
        day_of_week_one_hot = f.one_hot(day_of_week.to(torch.long), num_classes=7)

        temporal_features = torch.cat(
            (hour_of_day_one_hot, day_of_week_one_hot), dim=-1
        ).to(torch.float32)
        return temporal_features

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.train_mean) / self.train_std

    def inmemory_collate_fn(self, batch: Iterable[Data]):
        """
        Split the batch into tensors corresponding to the predictor (x) and the target (y) variables.

        To be called when the data is loaded from the memory.
        :param batch: A batch consisting of an iterable of Data points.
        :return: A tuple containing the x and y tensors.
        """
        x, y = zip(*batch)

        x_signal = torch.stack([self.normalize(x_[..., 0]) for x_ in x]).unsqueeze(
            dim=-1
        )
        y_signal = torch.stack([y_[..., 0] for y_ in y]).unsqueeze(
            dim=-1)

        x_temporal = torch.stack([self.onehot_temporal(x_[..., 1:]) for x_ in x])
        y_temporal = torch.stack([self.onehot_temporal(y_[..., 1:]) for y_ in y])

        return x_signal, y_signal, x_temporal, y_temporal

    def ondisk_collate_fn(self, batch: Iterable[Data]):
        """
        Splits the batch into tensors corresponding to the predictor (x) and the target (y) variables.

        To be called when the data is loaded from the disk memory.
        :param batch: A batch consisting of an iterable of Data points.
        :return: A tuple containing the x and y tensors.
        """
        x_signal = torch.stack([self.normalize(data.x[..., 0]) for data in batch])
        y_signal = torch.stack([data.y[..., 0] for data in batch])

        x_temporal = torch.stack(
            [self.onehot_temporal(data.x[..., 1:]) for data in batch]
        )
        y_temporal = torch.stack(
            [self.onehot_temporal(data.y[..., 1:]) for data in batch]
        )

        return x_signal, y_signal, x_temporal, y_temporal

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create a Dataloader over the training data.

        :return: The created dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create a Dataloader over the validation data.

        :return: The created dataloader.
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create a Dataloader over the test data.

        :return: The created dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            collate_fn=self.collate_fn,
        )

    def get_normalizers(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.train_mean, self.train_std
