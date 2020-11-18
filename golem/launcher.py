import hydra
import os
from hydra import utils as hydra_utils
from omegaconf import DictConfig
from golem import utils
import golem as glm
from distutils.dir_util import copy_tree
from shutil import copy2

def launch(init, entry_points, scripts={}):
    entry_point = os.environ.get("GOLEM_ENTRY_POINT", None)
    logger = utils.default_logger
    def launch_hydra(cfg: DictConfig):

        if entry_point is None or entry_point == "":
            script = os.environ["GOLEM_SCRIPT"]
            logger.info("Launching script: " + script) 
            scripts[script](cfg)
            return

        # This function fires when Hydra has initialized
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
