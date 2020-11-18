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
    parser = ArgumentParser(description="Golem CLI"
                            )
    default_experiment_id = now.strftime("%d-%m-%Y-%H:%M:%S")
    parser.add_argument("--experiment_id", default=default_experiment_id, type=str,
                        help="Experiment ID"
                        )
    parser.add_argument("--force_sync_glm", default=False, action='store_true',
                        help="Force sync glm repo to all nodes"
                        )
    parser.add_argument("--restore", default=False, action='store_true',
                        help="Restore from the experiment_id"
                        )
    parser.add_argument("--disable_auto_revive", default=False, action='store_true',
                        help="Disable the autorevive feature of Golem"
                        )
    parser.add_argument("network_config", type=str,
                        help="Network config (path to yaml file with the nodes)"
                        )
    parser.add_argument("job_name", type=str,
                        help="Job name"
                        )
    parser.add_argument("job_config", type=str,
                        help="Config name"
                        )
    return parser.parse_args()


def fit_job_on_nodes(current_rank, current_worldsize, placement, task, config, force_node=None):
    remaining_process_to_fit = config['processes']
    requirements = config['requirements']
    required_cpus = requirements.get('cpus', 0)
    required_gpus = requirements.get('gpus', 0)
    required_slow_gpus = requirements.get('slow_gpus', 0)
    for k, node in placement.items():
        rcpus = node['remaining_cpus']
        rgpus = node['remaining_gpus']
        rslow_gpus = node['remaining_slow_gpus']
        if force_node is not None and force_node != k and remaining_process_to_fit == config['processes']:
            continue
        if node['remaining_cpus'] >= required_cpus and node['remaining_gpus'] >= required_gpus and node['remaining_slow_gpus'] >= required_slow_gpus:
            # Compute how many processes we can fit
            processes = min(rcpus // required_cpus if required_cpus > 0 else 1e10, rgpus //
                            required_gpus if required_gpus > 0 else 1e10, rslow_gpus // required_slow_gpus if required_slow_gpus > 0 else 1e10,  remaining_process_to_fit)
            if processes > 0:
                remaining_process_to_fit -= processes
                node['remaining_cpus'] -= required_cpus * processes
                node['remaining_gpus'] -= required_gpus * processes
                node['remaining_slow_gpus'] -= required_slow_gpus * processes
                node['tasks'].append({
                    'task': task,
                    'start_rank': current_rank,
                    'processes': processes
                })
                current_rank += processes
                current_worldsize += processes
    if remaining_process_to_fit > 0:
        raise Exception('Could not fit task=' + task)
    return current_rank, current_worldsize, placement


def run_glm_launcher(c, experiment_id, job, config, tasks, world_size, master_ip, master_port, python_binary, daemon=True, restore=False):
    processes = tasks['processes']
    start_rank = tasks['start_rank']
    task = tasks['task']
    DAEMON_FLAG = ' --daemon'
    golem_path = c.run(
        '{} -c "import golem as glm;import os; print(os.path.abspath(os.path.join(glm.__path__[0], \'..\')))"'.format(python_binary), hide=True).stdout.strip()
    command = f'cd {golem_path} && {python_binary} -m golem.launcher_cli' + (DAEMON_FLAG if daemon else '') + f' --experiment_id {experiment_id} --master_addr {master_ip}' \
        + f' --master_port {master_port} --nproc {processes} --current_rank {start_rank} --world_size {world_size}' + (' --restore' if restore else '') + f' {job} {config} {task}'
    print("Running:", command)
    c.run(command)


def upload_job_folder_to_node(c, job, python_binary):
    job_folder_path = os.path.abspath(
        os.path.join(glm.__path__[0], '..', 'jobs', job))
    golem_path = c.run(
        '{} -c "import golem as glm;import os; print(os.path.abspath(os.path.join(glm.__path__[0], \'..\')))"'.format(python_binary), hide=True).stdout.strip()
    print('Uploading job folder...')
    patchwork.transfers.rsync(
        c, job_folder_path, golem_path + '/jobs/', exclude='__pycache__')


def upload_golem_to_node(c, python_binary):
    glm_path = os.path.abspath(
        os.path.join(glm.__path__[0], '..', 'golem'))
    golem_path = c.run(
        '{} -c "import golem as glm;import os; print(os.path.abspath(os.path.join(glm.__path__[0], \'..\')))"'.format(python_binary), hide=True).stdout.strip()
    print('Uploading golem to node...')
    patchwork.transfers.rsync(
        c, glm_path, golem_path + '/', exclude='__pycache__')
    
def find_rank_0_task(tasks):
    for t in tasks:
        if t['start_rank'] == 0:
            return t
    raise Exception(f'Could not find rank 0 in {tasks}')


def main_run(args, network_config, placement, experiment_id, job, job_config_name, world_size, master_node_ip, master_node_port, master_node, restore, upload=True):
    try:
        for node, host in zip(network_config['nodes'].keys(), list(map(lambda n: {'host': n['host'], 'user': n['user']}, network_config['nodes'].values()))):
            if len(placement[node]['tasks']) == 0:
                continue
            c = Connection(**host)
            print('Running Golem Launcher on node', node)
            for t in placement[node]['tasks']:
                if t['start_rank'] == 0:
                    print('Skipping.. It is rank 0!')
                    # We don't run rank 0 yet
                    continue
                python_binary = network_config['nodes'][node]['python_binary']
                if args.force_sync_glm and upload:
                    upload_golem_to_node(c, python_binary)
                if upload:
                    upload_job_folder_to_node(c, job, python_binary)
                run_glm_launcher(
                    c, experiment_id, job, job_config_name, t, world_size, master_node_ip, master_node_port, python_binary, restore=restore)
                print("Done!")
        master_node_config = network_config['nodes'][master_node]
        c_master = Connection(
            host=master_node_config['host'], user=master_node_config['user'])
        rank_0_task = find_rank_0_task(placement[master_node]['tasks'])
        try:
            # 3) Tail the logs of rank=0 on the master machine
            python_binary = network_config['nodes'][master_node]['python_binary']
            print(f'Running Golem Launcer on {master_node}, the master node')
            if args.force_sync_glm and upload:
                upload_golem_to_node(c_master, python_binary)
            if upload:
                upload_job_folder_to_node(c_master, job, python_binary)
            run_glm_launcher(c_master, experiment_id, job, job_config_name, rank_0_task, world_size,
                             master_node_ip, master_node_port, python_binary, daemon=False, restore=restore)
        except KeyboardInterrupt:
            print('KeyboardInterrupt!')
            for node, host in zip(network_config['nodes'].keys(), list(map(lambda n: {'host': n['host'], 'user': n['user']}, network_config['nodes'].values()))):
                if len(placement[node]['tasks']) == 0:
                    print('Skipping node', node)
                    continue
                c = Connection(**host)
                print('Cleaning zombie processes on node', node)
                python_binary = network_config['nodes'][node]['python_binary']
                pid_loc = f'~/golem/{job}/{experiment_id}/pids'
                c.run(
                    "{} -m golem.kill_pids {}".format(python_binary, pid_loc))

            sys.exit(0)
    except Exception as e:
        print("Error!", e)
        print("Cleaning...")
        for node, host in zip(network_config['nodes'].keys(), list(map(lambda n: {'host': n['host'], 'user': n['user']}, network_config['nodes'].values()))):
            if len(placement[node]['tasks']) == 0:
                print('Skipping node', node)
                continue
            c = Connection(**host)
            print('Cleaning zombie processes on node', node)
            python_binary = network_config['nodes'][node]['python_binary']
            pid_loc = f'~/golem/{job}/{experiment_id}/pids'
            c.run(
                "{} -m golem.kill_pids {}".format(python_binary, pid_loc))
        if not args.disable_auto_revive:
            print("Reviving in 5 seconds...")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                print("Revive aborted. Exiting.")
                sys.exit(0)
            main_run(args, network_config, placement, experiment_id, job, job_config_name, world_size, master_node_ip, master_node_port, master_node, restore=True, upload=False)
        else:
            sys.exit(0)

def main():
    args = parse_args()
    print("===GOLEM CLI===")
    experiment_id = args.experiment_id + '-' + args.job_config
    print(f"Experiment ID: {experiment_id}")
    restore = False
    if args.restore:
        print("Restoring!")
        restore = True
    job = args.job_name
    job_config_name = args.job_config
    full_network_config_path = os.path.join(os.getcwd(), args.network_config)
    print("Loading network config file from: ", full_network_config_path)
    full_job_config_file = os.path.abspath(os.path.join(golem.__path__[0], '..',
                                                        'jobs', args.job_name, '{}.yaml'.format(args.job_config)))
    print("Loading job config file from:", full_job_config_file)
    network_config = load_config_file(full_network_config_path)
    job_config = load_config_file(full_job_config_file)
    # 1) Determine the placement for each job based on the requirements
    placement = {}
    for node, config in network_config["nodes"].items():
        placement[node] = {
            'tasks': [],
            'remaining_cpus': config['hardware'].get('cpus', 0),
            'remaining_gpus': config['hardware'].get('gpus', 0),
            'remaining_slow_gpus': config['hardware'].get('slow_gpus', 0),
            'host': config['host'],
            'user': config['user']
        }
    master_node_ip = network_config['master_node']['ip']
    master_node_port = network_config['master_node']['port']
    master_node = network_config['master_node']['name']
    force_rank_0 = job_config['run']['force_rank_0']
    current_rank = 0
    current_worldsize = 0
    rank_0_config = job_config['run']['tasks'][force_rank_0]
    current_rank, current_worldsize, placement = fit_job_on_nodes(
        current_rank, current_worldsize, placement, force_rank_0, rank_0_config, force_node=master_node)
    for task, config in job_config['run']['tasks'].items():
        if task == force_rank_0:
            continue
        current_rank, current_worldsize, placement = fit_job_on_nodes(
            current_rank, current_worldsize, placement, task, config)
    print(
        f'Launching {current_worldsize} processes on {len(placement)} different nodes')
    print('--Placement--')
    print(yaml.dump(placement, allow_unicode=True, default_flow_style=False))
    # 2) Start all those jobs
    main_run(args=args, network_config=network_config, placement=placement, experiment_id=experiment_id,
            job=job, job_config_name=job_config_name, world_size=current_worldsize,
            master_node_ip=master_node_ip, master_node_port=master_node_port,
            master_node=master_node, restore=restore)

if __name__ == "__main__":
    main()
