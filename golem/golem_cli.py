import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
import yaml
import golem
from fabric import Connection


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


def fit_job_on_nodes(current_rank, current_worldsize, placement, task, config):
    remaining_process_to_fit = config['processes']
    requirements = config['requirements']
    required_cpus = requirements.get('cpus', 0)
    required_gpus = requirements.get('gpus', 0)
    for k, node in placement.items():
        rcpus = node['remaining_cpus']
        rgpus = node['remaining_gpus']
        if node['remaining_cpus'] >= required_cpus and node['remaining_gpus'] >= required_gpus:
            # Compute how many processes we can fit
            processes = min(rcpus // required_cpus if required_cpus > 0 else 1e10, rgpus //
                            required_gpus if required_gpus > 0 else 1e10, remaining_process_to_fit)
            if processes > 0:
                remaining_process_to_fit -= processes
                node['remaining_cpus'] -= required_cpus * processes
                node['remaining_gpus'] -= required_gpus * processes
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


def run_glm_launcher(c, job, config, tasks, world_size, master_ip, master_port, python_binary):
    print(tasks)
    processes = tasks['processes']
    start_rank = tasks['start_rank']
    task = tasks['task']
    golem_path = c.run(
        '{} -c "import golem as glm;import os; print(os.path.abspath(os.path.join(glm.__path__[0], \'..\')))"'.format(python_binary), hide=True).stdout.strip()
    c.run(f'cd {golem_path} && {python_binary} -m golem.launcher_cli --daemon --master_addr {master_ip}'
          f' --master_port {master_port} --nproc {processes} --current_rank {start_rank} --world_size {world_size} {job} {config} {task}')


def main():
    args = parse_args()
    print("GOLEM CLI")
    job = args.job_name
    job_config_name = args.job_config
    full_network_config_path = os.path.join(os.getcwd(), args.network_config)
    print("Loading network config file from: ", full_network_config_path)
    full_job_config_file = os.path.abspath(os.path.join(golem.__path__[0], '..',
                                                        'jobs', args.job_name, '{}.yaml'.format(args.job_config)))
    print("Loading job config file from:", full_job_config_file)
    network_config = load_config_file(full_network_config_path)
    job_config = load_config_file(full_job_config_file)
    print(network_config)
    print(job_config)
    # 1) Determine the placement for each job based on the requirements
    placement = {}
    for node, config in network_config["nodes"].items():
        placement[node] = {
            'tasks': [],
            'remaining_cpus': config['hardware'].get('cpus', 0),
            'remaining_gpus': config['hardware'].get('gpus', 0),
            'host': config['host'],
            'user': config['user']
        }
    force_rank_0 = job_config['run']['force_rank_0']
    current_rank = 0
    current_worldsize = 0
    rank_0_config = job_config['run']['tasks'][force_rank_0]
    current_rank, current_worldsize, placement = fit_job_on_nodes(
        current_rank, current_worldsize, placement, force_rank_0, rank_0_config)
    for task, config in job_config['run']['tasks'].items():
        if task == force_rank_0:
            continue
        current_rank, current_worldsize, placement = fit_job_on_nodes(
            current_rank, current_worldsize, placement, task, config)
    print(placement)
    print(current_rank)
    print(current_worldsize)
    master_node_ip = network_config['master_node']['ip']
    master_node_port = network_config['master_node']['port']
    # 2) Start all those jobs
    # TODO: Do this in parallel
    for node, host in zip(network_config['nodes'].keys(), list(map(lambda n: {'host': n['host'], 'user': n['user']}, network_config['nodes'].values()))):
        c = Connection(**host)
        print('Running jobs on node', node)
        for t in placement[node]['tasks']:
            # TODO: Job sync using scp
            run_glm_launcher(
                c, job, job_config_name, t, current_worldsize, master_node_ip, master_node_port, network_config['nodes'][node]['python_binary'])
            print("Done!")
    # 3) Tail the logs of rank=0 on the master machine


if __name__ == "__main__":
    main()
