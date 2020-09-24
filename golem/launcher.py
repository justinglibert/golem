import hydra
import os
from hydra import utils
from omegaconf import DictConfig


def launch(init, entry_points):
    entry_point = os.environ["GOLEM_ENTRY_POINT"]

    def launch_hydra(cfg: DictConfig):
        print(cfg)
        print("Current working directory  : {}".format(os.getcwd()))
        print("Original working directory : {}".format(utils.get_original_cwd()))
        print("to_absolute_path('foo')    : {}".format(
            utils.to_absolute_path("foo")))
        print("to_absolute_path('/foo')   : {}".format(utils.to_absolute_path("/foo")))
        init_tuple = init(cfg)
        entry_points[entry_point](init_tuple, cfg)
    return launch_hydra
