import copy
from typing import Union, Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType
from env import action_space_builder

from util.file_loader import Instance
import matplotlib.pyplot as plt


class JobEnv(gym.Env):
    def __init__(self, instance: Instance):
        self.instance = instance

        self.job_size = instance.job_size
        self.machine_size = instance.machine_size
        self.job_machine_nos = instance.machine_nos
        self.initial_process_time_channel = self.instance.processing_time
        self.max_process_time = np.max(instance.processing_time)
        self.total_working_time = np.sum(instance.processing_time)

        self.last_process_time_channel = None
        self.last_schedule_finish_channel = None

        obs = self.reset()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs.shape)
        self.action_space = gym.spaces.Discrete(18)
        self.action_choices = action_space_builder.build_action_choices(instance.processing_time)

        self.step_count = 0
        # 用于实时绘图
        self.process_time_channel = None
        self.schedule_finish_channel = None
        self.machine_utilization_channel = None
        self.i = None
        self.j = None
        self.cell_colors = self.build_cell_colors()

        # 用于tensorboard记录
        self.make_span = None

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.step_count = 0
        # 处理时间
        process_time_channel = copy.deepcopy(self.instance.processing_time)
        # 调度完成时
        schedule_finish_channel = np.zeros_like(process_time_channel)
        # 机器利用率
        machine_utilization_channel = np.zeros_like(process_time_channel)
        obs = self.get_obs(process_time_channel, schedule_finish_channel, machine_utilization_channel)
        return obs

    def step(
            self, action: ActType
    ) -> Union[Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]]:
        self.step_count += 1
        rule = self.action_choices[action]
        i, j = rule(self.last_process_time_channel)
        process_time_channel = copy.deepcopy(self.last_process_time_channel)
        schedule_finish_channel = copy.deepcopy(self.last_schedule_finish_channel)
        schedule_finish_channel[i, j] = (
            process_time_channel[i, j] if j == 0 else np.sum(process_time_channel[i, j - 1: j + 1])
        )
        process_time_channel[i, j] = 0

        machine_running_time_table = self.initial_process_time_channel - process_time_channel
        machine_utilization_channel = self.compute_machine_utilization(machine_running_time_table)

        obs = self.get_obs(process_time_channel, schedule_finish_channel, machine_utilization_channel)
        reward = self.compute_reward(schedule_finish_channel)
        done = np.sum(process_time_channel) == 0 or self.step_count >= 1000
        info = {"status": "timeout"} if self.step_count >= 1000 else {}

        return obs, reward, done, info

    def get_obs(self, process_time_channel, schedule_finish_channel, machine_utilization_channel):
        obs = np.array(
            [
                self.normalize_process_time_channel(process_time_channel),
                self.normalize_schedule_finish_channel(schedule_finish_channel),
                machine_utilization_channel,
            ],
            dtype=np.float32,
        )
        # obs = obs.swapaxes(0, 2)
        self.last_process_time_channel = process_time_channel
        self.last_schedule_finish_channel = schedule_finish_channel
        return obs

    def compute_reward(self, schedule_finish_channel):
        maxes = np.max(schedule_finish_channel, axis=1)
        self.make_span = np.max(maxes)
        return self.total_working_time / (self.machine_size * self.make_span)

    def normalize_process_time_channel(self, process_time_channel):
        return process_time_channel / self.max_process_time

    @staticmethod
    def normalize_schedule_finish_channel(schedule_finish_channel):
        maxes = np.max(schedule_finish_channel, axis=1)
        make_span = np.max(maxes)
        return schedule_finish_channel / make_span if make_span != 0 else schedule_finish_channel

    @staticmethod
    def compute_machine_utilization(machine_running_time_table):
        sums = np.sum(machine_running_time_table, axis=1)
        return machine_running_time_table / np.max(sums)

    def set_data_for_visualization(
            self, process_time_channel, schedule_finish_channel, machine_utilization_channel, i, j
    ):
        self.process_time_channel = process_time_channel
        self.schedule_finish_channel = schedule_finish_channel
        self.machine_utilization_channel = machine_utilization_channel
        self.i = i
        self.j = j

    def render(self, mode="human"):
        cell_colors = copy.deepcopy(self.cell_colors)
        cell_colors[self.i][self.j] = "#ff0521"
        ax1 = plt.subplot(1, 3, 1)
        ax1.table(cellText=self.process_time_channel, loc="center", cellColor=cell_colors)
        ax2 = plt.subplot(1, 3, 1)
        ax2.table(cellText=self.schedule_finish_channel, loc="center", cellColor=cell_colors)
        ax3 = plt.subplot(1, 3, 1)
        ax3.table(cellText=self.machine_utilization_channel, loc="center", cellColor=cell_colors)

        plt.draw()

    def build_cell_colors(self):
        cell_colors = []
        for i in range(self.job_size):
            colors = []
            for j in range(self.machine_size):
                colors.append("#ffffff")
            cell_colors.append(colors)
        return cell_colors
