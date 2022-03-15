import logging
from collections import Iterable
from typing import Optional, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Data, Dataset

logger = logging.getLogger('traffic')


class MetrLaDataModule(pl.LightningDataModule):
    """A data module that holds the dataloaders for the MetrLa dataset."""

    def __init__(self,
                 opt: dict,
                 dataset: Dataset) -> None:
        """
        Initialize the DataModule.
        :param opt: A dictionary of options.
        :param dataset: The dataset object.
        """
        super(MetrLaDataModule, self).__init__()

        self.opt = opt
        self.dataset = dataset
        self.train_batch_size = opt['train_batch_size']
        self.test_batch_size = opt['test_batch_size']

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.collate_fn: Optional[Callable] = None
        self.setup()

    def sample(self, indices, stage):
        sample = 'sample_dataset'
        if self.opt[sample]:
            sample_factor = f'sample_{stage}_factor'
            factor = self.opt[sample_factor]
            rng = np.random.default_rng(21)
            indices = rng.choice(indices, int(float(factor) * len(indices)), replace=False)

        return indices

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the DataModule.

        :param stage: Unused.
        :return: None.
        """
        dataset_len = len(self.dataset)
        valid_percentage = self.opt['valid_percentage']
        test_percentage = self.opt['test_percentage']
        train_percentage = 1. - (test_percentage + valid_percentage)

        last_train_index = int(train_percentage * dataset_len)
        last_valid_index = int((train_percentage + valid_percentage) * dataset_len)

        train_indices = range(0, last_train_index)
        valid_indices = range(last_train_index, last_valid_index)
        test_indices = range(last_valid_index, dataset_len)

        train_indices = self.sample(train_indices, 'train')
        valid_indices = self.sample(valid_indices, 'valid')
        test_indices = self.sample(test_indices, 'test')

        logger.debug(f"Samples: {len(train_indices)}/{len(valid_indices)}/{len(test_indices)}")
        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        inmemory_data = self.opt.get('in_memory')

        self.collate_fn = self.inmemory_collate_fn if inmemory_data else self.ondisk_collate_fn

    @staticmethod
    def inmemory_collate_fn(batch: Iterable[Data]):
        """
        Splits the batch into tensors corresponding to the predictor (x) and the target (y) variables.

        To be called when the data is loaded from the memory.
        :param batch: A batch consisting of an iterable of Data points.
        :return: A tuple containing the x and y tensors.
        """
        x, y = zip(*batch)

        x_signal = torch.stack([x_[..., 0] for x_ in x]).unsqueeze(dim=-1)
        y_signal = torch.stack([y_[..., 0] for y_ in y]).unsqueeze(dim=-1)

        x_temporal = torch.stack([x_[..., 1:] for x_ in x])
        y_temporal = torch.stack([y_[..., 1:] for y_ in y])

        return x_signal, y_signal, x_temporal, y_temporal

    @staticmethod
    def ondisk_collate_fn(batch: Iterable[Data]):
        """
        Splits the batch into tensors corresponding to the predictor (x) and the target (y) variables.

        To be called when the data is loaded from the disk memory.
        :param batch: A batch consisting of an iterable of Data points.
        :return: A tuple containing the x and y tensors.
        """
        x = torch.stack([data.x for data in batch])
        y = torch.stack([data.y for data in batch])
        return x, y

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create a Dataloader over the training data.

        :return: The created dataloader.
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=2,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create a Dataloader over the validation data.

        :return: The created dataloader.
        """
        return DataLoader(self.valid_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=2,
                          collate_fn=self.collate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create a Dataloader over the test data.

        :return: The created dataloader.
        """
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=2,
                          collate_fn=self.collate_fn)
