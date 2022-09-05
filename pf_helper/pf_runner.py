import json
import logging
import os
from queue import Queue

import numpy as np
from tqdm import tqdm

from envs.util.robot_status import Status
from util.plotters import mat_plot_util


class PfRunner:
    def __init__(self, args, model_dir: str):
        self.args = args
        self.gui = args.gui
        self.env = args.env
        self.writer = args.writer
        self.logger = args.logger
        self.train_config = args.train_config
        self.env_config = args.env_config
        self.val_frequency = args.val_frequency
        self.model_dir = model_dir
        if args.best:
            self.model_dir = os.path.join(self.model_dir, "best")
        elif args.epi is not None:
            self.model_dir = os.path.join(self.model_dir, "epi_{}".format(args.epi))

    def train(self, agent):
        if self.args.resume_dir is not None:
            logging.info("### Train resumes, loading model from: {}".format(self.args.resume_dir))
            agent.load(self.args.resume_dir, self.args.device)

        best_val_success_rate = 0
        status_queue = Queue(maxsize=100)
        for i in range(1, self.train_config["train"]["epochs"] + 1):
            # if i == 2:
            #     for model in models:
            #         for param in model.parameters():
            #             param.requires_grad = True
            #
            # a = models[0].state_dict()

            phase = "train"
            obs = self.env.reset()
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                action = agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                R += reward
                t += 1
                agent.observe(obs, reward, done, False)
                if done:
                    if status_queue.full():
                        status_queue.get()
                    status_queue.put(info["status"])
                    self.log_episode_status(phase, i, info)
                    break
            # if self.gui:
            #     self.plot_velocity()

            statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_steps", self.args.sim_env.episode_step_count, i)
            self.add_scalar(phase + "/episode_reward", R, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/loss", statistics[1][1], i)
            self.log_latest_status(i, status_queue, phase)

            if i % 10 == 0:
                self.logger.info("episode: {}, episode reward: {}, info: {}.".format(i, R, info))
                # save model
                agent.save(self.model_dir)
            if i > 0 and i % self.val_frequency == 0:
                success_rate = self.validate(agent, start_i=i // self.val_frequency)
                # agent.save("{}/epi_{}".format(self.model_dir, i))

                # save model with best val performance
                if success_rate >= best_val_success_rate:
                    best_val_success_rate = success_rate
                    agent.save(os.path.join(self.model_dir, "best"))
                agent.save(os.path.join(self.model_dir, "epi_{}".format(i)))

    def validate(self, agent, start_i=1):
        n_episodes = self.train_config["val"]["episodes"]
        with agent.eval_mode():
            success_num = 0
            collision_num = 0
            timeout_num = 0
            total_num = 0
            phase = "val"

            pbar = tqdm(range((start_i - 1) * n_episodes + 1, start_i * n_episodes + 1))
            for i in pbar:
                total_num += 1
                obs = self.env.reset()
                R = 0
                t = 0
                while True:
                    action = agent.act(obs)
                    obs, r, done, info = self.env.step(action)
                    R += r
                    t += 1
                    agent.observe(obs, r, done, False)
                    if done:
                        if info["status"] == Status.ReachGoal:
                            success_num += 1
                        elif info["status"] == Status.Timeout:
                            timeout_num += 1
                        elif info["status"] == Status.Collision:
                            collision_num += 1
                        break
                # if self.gui:
                #     self.plot_velocity()
                self.add_scalar(phase + "/episode_steps", self.args.sim_env.episode_step_count, i)
                self.add_scalar("{}/episode_reward".format(phase), R, i)
            self.add_scalar(phase + "/success", success_num, i)
            self.add_scalar(phase + "/timeout", timeout_num, i)
            self.add_scalar(phase + "/collision", collision_num, i)
            self.add_scalar("{}/success_rate".format(phase), success_num / total_num, i)
            self.add_scalar("{}/collision_rate".format(phase), collision_num / total_num, i)
            self.add_scalar("{}/timeout_rate".format(phase), timeout_num / total_num, i)
        return success_num / total_num

    def test(self, agent, load_model=False, start_i=1):
        # n_episodes = self.train_config["test"]["episodes"]
        n_episodes = self.args.test_episode
        navigation_time_on_success_episodes = {}
        history_actions = []
        if load_model:
            agent.load(self.model_dir, self.args.device)
        with agent.eval_mode():
            total_steps = 0
            success_num = 0
            collision_num = 0
            timeout_num = 0
            total_num = 0
            status_queue = Queue(maxsize=100)
            phase = "test"

            pbar = tqdm(range((start_i - 1) * n_episodes + 1, start_i * n_episodes + 1))
            for i in pbar:
                total_num += 1
                obs = self.env.reset()
                R = 0
                while True:
                    action = agent.act(obs)
                    obs, r, done, info = self.env.step(action)
                    if hasattr(self.env, "current_action"):
                        history_actions.append([self.env.current_action[0], self.env.current_action[1]])
                    R += r
                    # agent.observe(obs, r, done, False)
                    if done:
                        if status_queue.full():
                            status_queue.get()
                        status_queue.put(info["status"])

                        if info["status"] == Status.ReachGoal:
                            success_num += 1
                            self.add_scalar(phase + "/success", success_num, i)
                            navigation_time_on_success_episodes["{}".format(i)] = self.args.sim_env.episode_step_count
                        elif info["status"] == Status.Timeout:
                            timeout_num += 1
                            self.add_scalar(phase + "/timeout", timeout_num, i)
                        elif info["status"] == Status.Collision:
                            collision_num += 1
                            self.add_scalar(phase + "/collision", collision_num, i)
                        break

                self.add_scalar(phase + "/episode_steps", self.args.sim_env.episode_step_count, i)
                self.add_scalar("{}/episode_reward".format(phase), R, i)
                self.add_scalar("{}/average_success_rate".format(phase), success_num / total_num, i)
                self.log_latest_status(i, status_queue, phase)
                pbar.set_description("{}/average_success_rate:{}".format(phase, success_num / total_num))
                total_steps += self.args.sim_env.episode_step_count

                if self.gui:
                    self.plot_velocity(i)

            self.logger.info("Success num: {}, rate: {}".format(success_num, success_num / n_episodes))
            self.logger.info("Collision num: {}, rate: {}".format(collision_num, collision_num / n_episodes))
            self.logger.info("Timeout num: {}, rate: {}".format(timeout_num, timeout_num / n_episodes))
            self.logger.info("Average steps: {}".format(total_steps / n_episodes))
        # save success time to file
        dir_path = os.path.join(self.args.output, "test_result_new", self.args.test_scene)
        os.makedirs(dir_path, exist_ok=True)

        test_result = {
            "success": success_num,
            "success_rate": success_num / n_episodes,
            "collision": collision_num,
            "collision_rate": collision_num / n_episodes,
            "timeout": timeout_num,
            "timeout_rate": timeout_num / n_episodes,
            "navigation_time": navigation_time_on_success_episodes,
        }
        # store test result
        filename = os.path.join(
            dir_path,
            "c_{}+s_{}_speed_{}.".format(self.args.test_cuboid, self.args.test_static, self.args.test_velocity),
        )
        with open(filename + "json", "w") as f:
            json.dump(test_result, f)

        np.savez_compressed(filename + "npz", np.array(history_actions))

    def log_episode_status(self, phase, i, info):
        success, collide, timeout = 0, 0, 0
        if info["status"] == Status.ReachGoal:
            success = 1
        elif info["status"] == Status.Collision:
            collide = 1
        elif info["status"] == Status.Timeout:
            timeout = 1

        self.add_scalar(phase + "/success", success, i)
        self.add_scalar(phase + "/collision", collide, i)
        self.add_scalar(phase + "/timeout", timeout, i)

    def log_latest_status(self, i: int, status_queue: Queue, phase: str):
        """
        Add scalars for tensorboard
        """
        items = status_queue.queue
        collision_count = 0
        reach_goal_count = 0
        timeout_count = 0
        for item in items:
            if item == Status.Collision:
                collision_count += 1
            elif item == Status.ReachGoal:
                reach_goal_count += 1
            elif item == Status.Timeout:
                timeout_count += 1

        self.add_scalar("{}/latest_100_reach_goal_rate".format(phase), reach_goal_count / len(items), i)
        self.add_scalar("{}/latest_100_collision_rate".format(phase), collision_count / len(items), i)
        self.add_scalar("{}/latest_100_timeout_rate".format(phase), timeout_count / len(items), i)

    def add_scalar(self, tag: str, scalar_value, global_step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def plot_velocity(self, i):
        if hasattr(self.env, "robot"):
            robot = self.env.robot
        else:
            robot = self.env.sim_env.robot
        # mat_plot_util.plot_history_forward_angular_velocities(
        #     robot.history_v,
        #     robot.history_w,
        # )
        mat_plot_util.plot_history_forward_angular_velocities(
            history_v=robot.history_v,
            show=False,
            save_path=os.path.join(self.args.output, "applied_v", "{}.png".format(i)),
        )
        mat_plot_util.plot_history_forward_angular_velocities(
            history_v=robot.history_real_v,
            show=False,
            save_path=os.path.join(self.args.output, "real_v", "{}.png".format(i)),
        )
        mat_plot_util.plot_history_forward_angular_velocities(
            history_w=robot.history_w,
            show=False,
            save_path=os.path.join(self.args.output, "applied_w", "{}.png".format(i)),
        )
        mat_plot_util.plot_history_forward_angular_velocities(
            history_w=robot.history_real_w,
            show=False,
            save_path=os.path.join(self.args.output, "real_w", "{}.png".format(i)),
        )

        print(np.max(np.abs(np.array(robot.history_real_w))))
