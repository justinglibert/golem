import hydra
import os
from hydra import utils as hydra_utils
from omegaconf import DictConfig
from golem import utils
import golem as glm
from distutils.dir_util import copy_tree

def launch(init, entry_points):
    entry_point = os.environ["GOLEM_ENTRY_POINT"]
    logger = utils.default_logger

    def launch_hydra(cfg: DictConfig):
        logger.info("Rank: {}. World Size: {}. Golem Name: {}. Golem Entry Point: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["GOLEM_NAME"], os.environ["GOLEM_ENTRY_POINT"]))
        if int(os.environ["LOCAL_RANK"]) == 0:
            try:
                symlink = os.path.join(os.getcwd(), "..",  "latest")
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(os.getcwd(), symlink)
                logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                raise
            # Copy code into save folder
            glm_path = os.path.abspath(os.path.join(glm.__path__[0], '..'))
            copy_tree(glm_path + '/', os.getcwd() + '/code/' )
        init_tuple = init(cfg)
        entry_points[entry_point](init_tuple, cfg)
    return launch_hydra
