import logging
import os
from typing import Dict, List

import numpy as np
import torch
from tensorboardX import SummaryWriter

from env.job_env import JobEnv
from util import global_util


class PfRunner:
    def __init__(self, args, run_config: Dict, env):
        self.args = args
        self.run_config = run_config

        self.val_frequency = run_config["val"]["frequency"]
        self.render: JobEnv = args.render
        self.env = env
        self.writer = SummaryWriter(logdir=os.path.join(args.output))
        self.device = (
            torch.device("cuda:{}".format(args.gpu))
            if torch.cuda.is_available() and args.gpu >= 0
            else torch.device("cpu")
        )

        self.model_dir = os.path.join(args.output, args.model_dir)
        if args.epi is not None:
            self.model_dir = os.path.join(self.model_dir, "epi_{}".format(args.epi))

        self.vali_data = np.load(
            os.path.join(global_util.get_project_root(), "data", "generatedData{}_{}_Seed{}.npy").format(
                args.n_j, args.n_m, args.np_seed_validation
            )
        )

    def train(self, agent):
        shortest_make_span = np.inf
        for i in range(1, self.run_config["train"]["episodes"] + 1):
            phase = "train"
            obs = self.env.reset(episode=i, phase=phase)
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                action = agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                R += reward
                t += 1
                agent.observe(obs, reward, done, False)
                if done:
                    break
            if not self.args.render:
                self.env.render()

            statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_reward", R, i)
            self.add_scalar(phase + "/makespan", self.env.make_span, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/loss", statistics[1][1], i)
            self.add_scalar(phase + "/epsilon", agent.explorer.epsilon, i)

            if i % 10 == 0:
                logging.info("episode: {}, episode reward: {}, info: {}.".format(i, R, info))
                # save model
                agent.save(self.model_dir)
            if i > 0 and i % self.val_frequency == 0:
                make_span = self.validate(agent, start_i=i // self.val_frequency)

                # save model with best val performance
                if make_span < shortest_make_span:
                    shortest_make_span = make_span
                    agent.save(os.path.join(self.model_dir, "best"))
                agent.save(os.path.join(self.model_dir, "epi_{}".format(i)))

    def validate(self, agent, start_i=1, phase="val"):
        if phase == "test":
            agent.load(self.model_dir, self.device)

        n_episodes = len(self.vali_data)
        count = 0
        total_make_span = 0
        with agent.eval_mode():
            i = (start_i - 1) * n_episodes
            for data in self.vali_data:
                i += 1
                obs = self.env.reset(data=data, phase=phase)
                count += 1
                R = 0  # return (sum of rewards)
                t = 0  # time step
                while True:
                    action = agent.act(obs)
                    obs, reward, done, info = self.env.step(action)
                    R += reward
                    t += 1
                    if done:
                        break
                if self.args.render:
                    self.env.render()
                logging.info("makespan={}".format(self.env.make_span))
                statistics = agent.get_statistics()
                self.add_scalar(phase + "/episode_reward", R, i)
                self.add_scalar(phase + "/makespan", self.env.make_span, i)
                self.add_scalar(phase + "/average_q", statistics[0][1], i)
                self.add_scalar(phase + "/loss", statistics[1][1], i)
                total_make_span += self.env.make_span
        return total_make_span / n_episodes

    def add_scalar(self, tag: str, scalar_value, global_step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)
