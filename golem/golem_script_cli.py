import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
import yaml
import golem
import time
import numpy as np
from fabric import Connection
from datetime import datetime
import golem as glm
import patchwork.transfers

now = datetime.now()  # current date and time


def load_config_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Golem Script CLI"
                            )
    parser.add_argument("--experiment_id", type=str, required=True,
                        help="Experiment ID"
                        )
    parser.add_argument("job_name", type=str,
                        help="Job name"
                        )
    parser.add_argument("script", type=str,
                        help="Script name"
                        )
    return parser.parse_args()



def main():
    args = parse_args()
    print("===GOLEM SCRIPT CLI===")
    experiment_id = args.experiment_id
    print(f"Experiment ID: {experiment_id}")
    job = args.job_name
    home = os.path.expanduser("~")
    run_folder = home + '/golem/' + args.job_name + '/' + args.experiment_id

    current_env = os.environ.copy()
    current_env["RANK"] = str(0)
    current_env["LOCAL_RANK"] = str(0)
    current_env["GOLEM_SCRIPT"] = args.script
    current_env["GOLEM_NAME"] = '{}:{}'.format(
        args.script, 0)
    current_env["JOB_NAME"] = args.job_name
    current_env["GOLEM_EXPERIMENT_ID"] = args.experiment_id
    current_env["GOLEM_RUN_FOLDER"] = run_folder
    cmd = []
    cmd = [sys.executable, "-u"]
    cmd.append("-m")

    cmd.append('jobs.' + args.job_name + '.job')

    cmd.extend(['hydra.run.dir=' + run_folder,
                '--config-dir', run_folder + '/.hydra/', '--config-name', 'config'])
    process = subprocess.Popen(cmd, env=current_env)
    process.wait()

if __name__ == "__main__":
    main()
