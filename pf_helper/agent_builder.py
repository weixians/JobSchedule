import os

import numpy as np
import torch

from pfrl import agents, explorers, replay_buffers
from util import global_util


def build_dqn_agent(args, phi, config_dir: str, gpu: int, model: torch.nn.Module, action_size):
    agent_name = "dqn"
    agent_config = global_util.load_yaml(os.path.join(config_dir, "agent_config.yml"))

    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=agent_config["explorer"]["start_epsilon"],
        end_epsilon=agent_config["explorer"]["end_epsilon"],
        decay_steps=agent_config["explorer"]["decay_steps"],
        random_action_func=lambda: np.random.randint(action_size),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=agent_config["adam"]["lr"],
        eps=agent_config["adam"]["lr_eps"],
        weight_decay=agent_config["adam"]["weight_decay"],
        amsgrad=agent_config["adam"]["amsgrad"],
    )

    replay_buffer = replay_buffers.PrioritizedReplayBuffer(
        capacity=agent_config["prioritized_replay_buffer"]["capacity"],
        num_steps=agent_config["prioritized_replay_buffer"]["num_steps"],
        alpha=agent_config["prioritized_replay_buffer"]["alpha"],
        beta0=agent_config["prioritized_replay_buffer"]["beta0"],
        betasteps=agent_config["prioritized_replay_buffer"]["betasteps"],
        normalize_by_max=agent_config["prioritized_replay_buffer"]["normalize_by_max"],
    )

    episodic_update_len = args.train_config["episodic_replay_buffer"]["num_steps"] if args.recurrent else None
    # pfrl.nn.to_factorized_noisy(model, sigma_scale=0.1)

    # Now create an agent that will interact with the environment.
    agent = agents.DoubleDQN(
        model,
        optimizer,
        replay_buffer=replay_buffer,
        gamma=agent_config[agent_name]["gamma"],
        explorer=explorer,
        replay_start_size=agent_config[agent_name]["replay_start_size"],
        target_update_interval=agent_config[agent_name]["target_update_interval"],
        update_interval=agent_config[agent_name]["update_interval"],
        target_update_method=agent_config[agent_name]["target_update_method"],
        phi=phi,
        gpu=gpu,
        recurrent=args.recurrent and not low_level,
        episodic_update_len=episodic_update_len,
    )
    return agent
