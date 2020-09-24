import hydra
from omegaconf import DictConfig
from typing import Tuple
import golem as glm


def learner(init: Tuple, cfg: DictConfig):
    logger = glm.utils.default_logger
    logger.info("I am the learner")


def actor(init: Tuple, cfg: DictConfig):
    logger = glm.utils.default_logger
    logger.info("I am the actor")


def init(cfg: DictConfig):
    logger = glm.utils.default_logger
    logger.info("I am the init")
    return (1, 2)


hydra.main()(glm.launcher.launch(init, {"actor": actor, "learner": learner}))()
