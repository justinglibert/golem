import hydra
from omegaconf import DictConfig
from typing import Tuple
import golem as glm


def learner(init: Tuple, cfg: DictConfig):
    print("I am a learner!")
    print(init)
    print(cfg)


def actor(init: Tuple, cfg: DictConfig):
    print("I am an actor!")
    print(init)
    print(cfg)


def init(cfg: DictConfig):
    print("I am the init")
    print(cfg)
    return (1, 2)


hydra.main()(glm.launcher.launch(init, {"actor": actor, "leaner": learner}))()
