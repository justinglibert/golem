import hydra
import os
from hydra import utils
from omegaconf import DictConfig


def launch(init, entry_points):
    print(init, entry_points)

    def launch_hydra(cfg: DictConfig):
        print(cfg)
        print("Current working directory  : {}".format(os.getcwd()))
        print("Original working directory : {}".format(utils.get_original_cwd()))
        print("to_absolute_path('foo')    : {}".format(
            utils.to_absolute_path("foo")))
        print("to_absolute_path('/foo')   : {}".format(utils.to_absolute_path("/foo")))
    return launch_hydra
