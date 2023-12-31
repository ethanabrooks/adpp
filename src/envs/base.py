from abc import ABC, abstractmethod

import gym
import gym.spaces


class Env(gym.Env, ABC):
    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Discrete:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def task_space(self) -> gym.Space:
        pass

    @abstractmethod
    def get_task(self, n: int = 1):
        pass
