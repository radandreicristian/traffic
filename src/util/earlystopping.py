import logging

import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Source: https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self,
                 checkpoint_path: str,
                 patience,
                 verbose=False,
                 delta=0) -> None:
        """

        :param patience: How long to wait after last time validation loss improved.
        :param verbose: If True, prints a message for each validation loss improvement.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param path: Path for the checkpoint to be saved to.

        :returns None.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.logger = logging.getLogger('traffic')

    def __call__(self,
                 val_loss: float,
                 model: torch.nn.Module) -> None:
        """
        Checks whether the early stopping should trigger.

        :param val_loss: The current validation loss.
        :param model: The model.
        :return: None.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.debug(f'Early stopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self,
                        val_loss,
                        model) -> None:
        """
        Save the model checkpoint to the path.

        :param val_loss: The current validation loss.
        :param model: The model.
        :return: None.
        """
        if self.verbose:
            self.logger.debug(f'Validation loss decreased ({self.val_loss_min:.2f} --> {val_loss:.2f}).  Saving model.')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
