import logging
from typing import Optional, Callable, Sequence, List, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset


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
        dataset_len = self.dataset.len()
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

        self.train_dataset = KeyedSubset(self.dataset, train_indices)

        train_features = torch.stack([x["features"][:, :] for x, _ in self.train_dataset])
        self.train_mean = torch.mean(train_features)
        self.train_std = torch.std(train_features)

        self.valid_dataset = KeyedSubset(self.dataset, valid_indices)
        self.test_dataset = KeyedSubset(self.dataset, test_indices)

    @staticmethod
    def onehot_day_of_week(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.long)

    @staticmethod
    def onehot_interval_of_day(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.long)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.train_mean) / self.train_std

    def collate_fn(
            self,
            batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
            ) -> [Dict[torch.Tensor, torch.Tensor]]:
        """
        Split the batch into tensors corresponding to the predictor (x) and the target (y) variables.

        Iterable of pairs -> Pair of iterables via zip
        Iterable of pairs of dicts -> Pair of dict of iterables via zip

        To be called when the data is loaded from the memory.
        :param batch: A batch consisting of an iterable of Data points.
        :return: A tuple containing the x and y tensors.
        """

        normalization_functions = {
            "features": self.normalize,
            "day_of_week": self.onehot_day_of_week,
            "interval_of_day": self.onehot_interval_of_day,
        }

        keys = list(batch[0][0].keys())
        x = {k: [] for k in keys}
        y = {k: [] for k in keys}
        x["raw_features"] = []
        y["raw_features"] = []
        # Sample is a tuple (x, y)
        for sample in batch:
            # x_, y_ are dictionaries of {'features': ...,}
            x_, y_ = sample
            for key in keys:
                x__, y__ = x_[key], y_[key]
                normalizer = normalization_functions.get(key, None)
                if normalizer:
                    x__ = normalizer(x__)
                    y__ = normalizer(y__)
                    x[key].append(x__)
                    y[key].append(y__)
            x["raw_features"].append(x_["features"])
            y["raw_features"].append(y_["features"])

        updated_keys = x.keys()
        for key in updated_keys:
            x[key] = torch.stack(x[key])
            y[key] = torch.stack(y[key])
            if key != "features" and key != "raw_features":
                x[key] = x[key].squeeze(dim=-1)
                y[key] = y[key].squeeze(dim=-1)
        return x, y

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
            pin_memory=True,
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


class KeyedSubset:
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset.get(self.indices[idx])

    def __len__(self):
        return len(self.indices)
