from collections import Iterable
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Subset, Dataset, DataLoader
from torch_geometric.data import Data


class MetrLaInMemoryDataModule(pl.LightningDataModule):
    def __init__(self,
                 opt: dict,
                 dataset):
        super(MetrLaInMemoryDataModule, self).__init__()

        self.opt = opt
        self.dataset = dataset
        self.train_batch_size = opt['train_batch_size']
        self.test_batch_size = opt['test_batch_size']

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_len = len(self.dataset)
        valid_percentage = self.opt['valid_percentage']
        test_percentage = self.opt['test_percentage']
        train_percentage = 1. - (test_percentage + valid_percentage)

        last_train_index = int(train_percentage * dataset_len)
        last_valid_index = int((train_percentage + valid_percentage) * dataset_len)

        train_indices = range(0, last_train_index)
        if self.opt['sample_train']:
            factor = self.opt['sample_train_factor']
            rng = np.random.default_rng(21)
            train_indices = rng.choice(train_indices, int(factor * len(train_indices)), replace=False)
        valid_indices = range(last_train_index, last_valid_index)
        test_indices = range(last_valid_index, dataset_len)
        print("n train samples", len(train_indices))
        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    @staticmethod
    def collate_fn(batch: Iterable[Data]):
        x, y = zip(*batch)
        x = torch.stack([x_ for x_ in x])
        y = torch.stack([y_ for y_ in y])
        return x, y

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=2,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=2,
                          collate_fn=self.collate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=2,
                          collate_fn=self.collate_fn)
