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
    parser.add_argument("--experiment_id", type=str,
                        help="The experiment id")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    parser.add_argument("--daemon", action="store_true",
                        help="Daemon mode")

    parser.add_argument("--restore", action="store_true",
                        help="Restore mode")
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

    print("==GOLEM LAUNCHER==")
    home = os.path.expanduser("~")
    run_folder = home + '/golem/' + args.job_name + '/' + args.experiment_id
    os.makedirs(run_folder, exist_ok=True)

    with open(run_folder + '/job', 'a') as file_object:
        file_object.write(args.job_name)

    for local_rank in range(0, args.nproc):
        # each process's rank
        dist_rank = args.current_rank + local_rank

        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["GOLEM_ENTRY_POINT"] = args.entry_point
        current_env["GOLEM_NAME"] = '{}:{}'.format(
            args.entry_point, dist_rank)
        current_env["JOB_NAME"] = args.job_name
        current_env["GOLEM_EXPERIMENT_ID"] = args.experiment_id
        current_env["GOLEM_RUN_FOLDER"] = run_folder
        if args.restore:
            current_env['GOLEM_RESTORE'] = str(1)

        # spawn the processes
        cmd = []
        cmd = [sys.executable, "-u"]
        cmd.append("-m")

        cmd.append('jobs.' + args.job_name + '.job')

        cmd.extend(['hydra.run.dir=' + run_folder,
                    '--config-dir', './jobs/' + args.job_name + '/', '--config-name', args.job_config])

        if args.daemon:
            stdout = open(
                run_folder + f'/{local_rank}-{args.entry_point}.stdout.logs', 'w+')
            stderr = open(
                run_folder + f'/{local_rank}-{args.entry_point}.stderr.logs', 'w+')
            process = subprocess.Popen(
                cmd, env=current_env, stdout=stdout, stderr=stderr)
            processes.append(process)
            print(
                f"Running process {local_rank}-{args.entry_point} as a deamon. pid={process.pid}")
        else:
            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)
        with open(run_folder + '/pids', "a") as file_object:
            file_object.write(str(process.pid) + '\n')

    if args.daemon:
        sys.exit(0)
        return
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    main()
