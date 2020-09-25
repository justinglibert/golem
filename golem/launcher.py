import hydra
import os
from hydra import utils as hydra_utils
from omegaconf import DictConfig
from golem import utils


def launch(init, entry_points):
    entry_point = os.environ["GOLEM_ENTRY_POINT"]
    logger = utils.default_logger

    def launch_hydra(cfg: DictConfig):
        logger.info("Rank: {}. World Size: {}. Golem Name: {}. Golem Entry Point: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["GOLEM_NAME"], os.environ["GOLEM_ENTRY_POINT"]))
        init_tuple = init(cfg)
        entry_points[entry_point](init_tuple, cfg)
    return launch_hydra
