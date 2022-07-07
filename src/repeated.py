import gc
import os

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.autoregressive_experiment import AutoregressiveExperiment


@hydra.main(config_path="conf", config_name="config_full_autoregressive.yaml")
def main(config: DictConfig):
    for i in range(10):
        seed_everything()
        experiment = AutoregressiveExperiment(config)
        experiment.run()

        # return experiment.get_test_rmse()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("triggered in main")
        torch.cuda.empty_cache()
        gc.collect()
        os.system("pkill -9 python")