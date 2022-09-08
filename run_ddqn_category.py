import os

from env.job_env import JobEnv

from pf_helper.agent_builder import build_dqn_agent
from pf_helper.pf_runner import PfRunner
from util import global_util
from util.file_loader import load_instances
from util.util import parse_args, copy_data_folder_to_output, build_network

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    args = parse_args()
    args.output = os.path.join(args.output, "{}_{}".format(args.net, "dueling") if args.dueling else args.net)

    global_util.setup_logger()
    copy_data_folder_to_output(args, True)
    run_config = global_util.load_yaml(os.path.join(args.output, "data/run_config.yml"))
    instance_dict = load_instances(os.path.join(args.output, "data/jobshop1.txt"))
    instance = instance_dict[run_config["instance"]]
    skip_num = max(1, int(run_config["skip_ratio"] * instance.job_size * instance.machine_size))

    instances = []
    for key, value in instance_dict.items():
        if value.machine_size == instance.machine_size and value.job_size == instance.job_size:
            instances.append(value)

    env = JobEnv(args, instance.job_size, instance.machine_size)
    input_dim = env.observation_space.shape
    action_size = env.action_space.n
    q_local_net = build_network(args, action_size)
    agent = build_dqn_agent(
        lambda x: x, args.data_dir, gpu=args.gpu, model=q_local_net, action_size=action_size, skip_num=skip_num
    )

    # train
    pf_runner = PfRunner(args, run_config, env, instances)
    if args.test:
        pf_runner.validate(agent, start_i=1, phase="test")
    else:
        pf_runner.train(agent)
