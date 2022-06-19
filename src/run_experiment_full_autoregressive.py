import gc
import os

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.autoregressive_experiment import AutoregressiveExperiment

seed_everything(21)


@hydra.main(config_path="conf", config_name="config_full_autoregressive.yaml")
def main(config: DictConfig):
    experiment = AutoregressiveExperiment(config)
    experiment.run()

    # Return the experiment test RMSE so that it can be integrated in an optuna hyperparameter search.
    return experiment.get_test_rmse()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("triggered in main")
        # Am I asking for too much when I am thinking that the Python process should close upon CTRL+C? Guess so.
        torch.cuda.empty_cache()
        gc.collect()
        os.system("pkill -9 python")
