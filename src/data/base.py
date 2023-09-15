from abc import ABC, abstractmethod
from functools import lru_cache

import torch
from torch.utils.data import Dataset

from envs.base import Env


class Data(Dataset, ABC):
    @property
    @abstractmethod
    def include_goal(self) -> bool:
        pass

    @property
    @abstractmethod
    def return_range(self) -> tuple[float, float]:
        pass

    @property
    @lru_cache
    def n_tokens(self):
        return 1 + self.data.max().round().long().item()

    @property
    @lru_cache
    def pad_value(self):
        return self.data.max().round().long().item()

    @property
    @abstractmethod
    def episodes_per_rollout(self):
        pass

    @property
    def step_dim(self):
        return sum(self._dims)

    @abstractmethod
    def build_env(self) -> Env:
        pass

    @abstractmethod
    def cat_sequence(self, goals, observations, actions, rewards):
        pass

    @abstractmethod
    def get_metrics(
        self, logits: torch.Tensor, graphs_per_component: int, sequence: torch.Tensor
    ):
        pass

    @abstractmethod
    def split_sequence(self, sequence: torch.Tensor):
        pass

    @lru_cache
    def weights(self, shape, **kwargs):
        weights = torch.ones(shape)
        sequence = self.split_sequence(weights)
        for k, v in kwargs.items():
            assert k in sequence, f"Invalid key {k}"
            sequence[k] *= v
        return self.cat_sequence(**sequence).cuda()
