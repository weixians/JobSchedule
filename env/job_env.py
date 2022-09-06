import copy
import os
from typing import Union, Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType
from env import action_space_builder

from util.file_loader import Instance
import matplotlib.pyplot as plt


class JobEnv(gym.Env):
    def __init__(self, args, instance: Instance):
        self.args = args
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

        self.make_span = None
        self.machine_finish_time = None
        # 用于记录
        self.episode_count = 0
        self.step_count = 0
        # 用于实时绘图
        self.process_time_channel = None
        self.schedule_finish_channel = None
        self.machine_utilization_channel = None
        self.i = None
        self.j = None
        self.cell_colors = self.build_cell_colors()

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.make_span = 0
        self.machine_finish_time = np.zeros(self.machine_size, dtype=np.int32)
        # 用于记录
        self.episode_count = kwargs.get("episode") if "episode" in kwargs else 0
        self.step_count = 0

        # 处理时间
        process_time_channel = copy.deepcopy(self.instance.processing_time)
        # 调度完成时
        schedule_finish_channel = np.zeros_like(process_time_channel)
        # 机器利用率
        machine_utilization_channel = np.zeros_like(process_time_channel)
        obs = self.get_obs(process_time_channel, schedule_finish_channel, machine_utilization_channel)
        self.set_data_for_visualization(
            process_time_channel, schedule_finish_channel, machine_utilization_channel, None, None
        )
        return obs

    def step(
        self, action: ActType
    ) -> Union[Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]]:
        self.step_count += 1
        rule = self.action_choices[action]
        i, j = rule(self.last_process_time_channel)
        process_time_channel = copy.deepcopy(self.last_process_time_channel)
        schedule_finish_channel = copy.deepcopy(self.last_schedule_finish_channel)
        schedule_finish_channel[i, j] = process_time_channel[i, j] + self.make_span
        process_time_channel[i, j] = 0
        machine_running_time_table = self.initial_process_time_channel - process_time_channel
        machine_utilization_channel = self.compute_machine_utilization(machine_running_time_table)

        self.compute_make_span(schedule_finish_channel, i, j)
        self.update_machine_finish_time(i, j)

        obs = self.get_obs(process_time_channel, schedule_finish_channel, machine_utilization_channel)
        reward = self.compute_reward()
        done = np.sum(process_time_channel) == 0
        self.set_data_for_visualization(
            process_time_channel, schedule_finish_channel, machine_utilization_channel, i, j
        )
        return obs, reward, done, {}

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

    def compute_reward(self):
        return self.total_working_time / (self.machine_size * self.make_span)

    def compute_make_span(self, schedule_finish_channel, i, j):
        # 获取当前job到目前为止的完成时间
        op_finish_time = 0 if j == 0 else schedule_finish_channel[i, j - 1]
        # 对比当前任务时间及新任务对应机器的完成时间，取大的
        self.make_span = (
            max(op_finish_time, self.machine_finish_time[self.job_machine_nos[i, j]])
            + self.initial_process_time_channel[i, j]
        )

    def update_machine_finish_time(self, i, j):
        self.machine_finish_time[self.job_machine_nos[i, j]] += self.initial_process_time_channel[i, j]

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
        self.machine_utilization_channel = np.around(machine_utilization_channel, decimals=2)
        self.i = i
        self.j = j

    def render(self, mode="human"):
        plt.clf()
        # plt.figure(figsize=(12, 3))
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
        plt.rcParams["figure.figsize"] = (16, 3)
        cell_colors = copy.deepcopy(self.cell_colors)
        if self.i is not None and self.j is not None:
            cell_colors[self.i][self.j] = "#ff0521"
        ax00 = plt.subplot(2, 3, 1)
        ax00.set_title("操作所需时间")
        ax00.table(cellText=self.initial_process_time_channel, loc="center", cellColours=cell_colors)
        ax00.axis("off")

        ax01 = plt.subplot(2, 3, 2)
        ax01.set_title("对应机器编号")
        ax01.table(cellText=self.job_machine_nos, loc="center", cellColours=cell_colors)
        ax01.axis("off")

        ax02 = plt.subplot(2, 3, 3)
        colors = ["#ffffff" for i in range(self.machine_size)]
        colors[self.job_machine_nos[self.i, self.j]] = "#ff0521"
        ax02.set_title("机器完成时间")
        ax02.table(cellText=[self.machine_finish_time], loc="center", cellColours=[colors])
        ax02.axis("off")

        ax1 = plt.subplot(2, 3, 4)
        ax1.set_title("processing time")
        ax1.table(cellText=self.process_time_channel, loc="center", cellColours=cell_colors)
        ax1.axis("off")

        ax2 = plt.subplot(2, 3, 5)
        ax2.set_title("schedule finish")
        ax2.table(cellText=self.schedule_finish_channel, loc="center", cellColours=cell_colors)
        ax2.axis("off")

        ax3 = plt.subplot(2, 3, 6)
        ax3.set_title("machine utilization")
        ax3.table(cellText=self.machine_utilization_channel, loc="center", cellColours=cell_colors)
        ax3.axis("off")

        folder = os.path.join(self.args.output, "render")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(
            os.path.join(folder, "e_{}_step_{}.png".format(self.episode_count, self.step_count)),
            bbox_inches="tight",
            pad_inches=0.5,
            dpi=500,
        )
        # plt.pause(0.4)

    def build_cell_colors(self):
        cell_colors = []
        for i in range(self.job_size):
            colors = []
            for j in range(self.machine_size):
                colors.append("#ffffff")
            cell_colors.append(colors)
        return cell_colors
