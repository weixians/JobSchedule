import json
import logging
import os
from queue import Queue
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

from env.job_env import JobEnv


class PfRunner:
    def __init__(self, args, run_config: Dict, env):
        self.args = args
        self.run_config = run_config

        self.val_frequency = run_config["val"]["frequency"]
        self.render: JobEnv = args.render
        self.env = env
        self.writer = SummaryWriter(logdir=os.path.join(args.output))

        self.model_dir = os.path.join(args.output, args.model_dir)
        # if args.best:
        #     self.model_dir = os.path.join(self.model_dir, "best")
        # elif args.epi is not None:
        #     self.model_dir = os.path.join(self.model_dir, "epi_{}".format(args.epi))

    def train(self, agent):
        # if self.args.resume_dir is not None:
        #     logging.info("### Train resumes, loading model from: {}".format(self.args.resume_dir))
        #     agent.load(self.args.resume_dir, self.args.device)

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
                if self.args.render:
                    self.env.render()
                if done:
                    break
            if not self.args.render:
                self.env.render()

            statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_reward", R, i)
            self.add_scalar(phase + "/makespan", self.env.make_span, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/loss", statistics[1][1], i)

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
        n_episodes = self.run_config["val"]["episodes"]
        total_make_span = 0
        with agent.eval_mode():
            pbar = tqdm(range((start_i - 1) * n_episodes + 1, start_i * n_episodes + 1))
            for i in pbar:
                obs = self.env.reset(episode=i, phase=phase)
                R = 0  # return (sum of rewards)
                t = 0  # time step
                # if self.args.render:
                #     plt.ion()
                while True:
                    action = agent.act(obs)
                    obs, reward, done, info = self.env.step(action)
                    R += reward
                    t += 1
                    agent.observe(obs, reward, done, False)
                    if self.args.render:
                        self.env.render()
                    if done:
                        break
                # plt.ioff()
                # plt.show()

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
