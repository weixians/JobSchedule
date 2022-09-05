from abc import ABC

import gym
import gym.spaces

from subgoal_agent import subgoal_helper


class DiscreteActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, args):
        env = args.env
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        # self.action_space = env.action_space

        subgoal_config = args.env_config["subgoal_space"]
        self.subgoal_space = subgoal_helper.get_subgoal_options(subgoal_config)

    def action(self, action):
        return self.subgoal_space[action]
