import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--current_rank", type=int, default=0,
                        help="The current number of processes that already have been initialized")
    parser.add_argument("--nproc", type=int, default=1,
                        help="Number of processes to launch on this node")
    parser.add_argument("--world_size", type=int, default=1,
                        help="The final world size")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")

    # positional
    parser.add_argument("job_name", type=str,
                        help="Job name"
                        )

    parser.add_argument("job_config", type=str,
                        help="Config name"
                        )
    parser.add_argument("entry_point", type=str,
                        help="Entry point"
                        )
    return parser.parse_args()


def main():
    args = parse_args()

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(args.world_size)

    processes = []

    print("GOLEM LAUNCHER")
    print("World Size", args.world_size)
    print("N proc", args.nproc)
    print("Current rank", args.current_rank)

    for local_rank in range(0, args.nproc):
        # each process's rank
        dist_rank = args.current_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["GOLEM_ENTRY_POINT"] = args.entry_point
        current_env["GOLEM_NAME"] = '{}:{}'.format(
            args.entry_point, dist_rank)

        # spawn the processes
        cmd = []
        cmd = [sys.executable, "-u"]
        cmd.append("-m")

        cmd.append('jobs.' + args.job_name + '.job')
        home = os.path.expanduser("~")
        cmd.extend(['hydra.run.dir=' + home + '/golem/' + args.job_name + '/${now:%Y-%m-%d}/${now:%H-%M-%S}',
                    '--config-dir', './jobs/' + args.job_name + '/', '--config-name', args.job_config])

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    main()
