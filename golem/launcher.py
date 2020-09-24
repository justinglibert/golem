import hydra
from omegaconf import DictConfig


def launch(init, entry_points):
    print(init, entry_points)

    def launch_hydra(cfg: DictConfig):
        print(cfg)
    return launch_hydra
