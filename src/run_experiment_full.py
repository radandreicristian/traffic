import gc
import os

import hydra
import torch
from omegaconf import DictConfig

from src.experiment import Experiment


@hydra.main(config_path="conf", config_name="config_full.yaml")
def main(config: DictConfig):
    experiment = Experiment(config)
    experiment.run()

    # Return the experiment test RMSE so that it can be integrated in an optuna hyperparameter search.
    return experiment.get_test_rmse()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Am I asking for too much when I am thinking that the Python process should close upon CTRL+C? Guess so.
        torch.cuda.empty_cache()
        gc.collect()
        os.system("pkill -9 python")
