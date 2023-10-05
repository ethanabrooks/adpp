from typing import Union

import torch
from gym.spaces import Discrete, MultiDiscrete

from envs.base import Env
from grid_world.grid_world import GridWorld


class Env(GridWorld, Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_tasks=1, **kwargs)
        self.current_state = None
        self.t = None

    @property
    def action_space(self):
        return Discrete(len(self.deltas))

    @property
    def observation_space(self):
        return MultiDiscrete([self.grid_size, self.grid_size])

    @property
    def task_space(self):
        return MultiDiscrete([self.grid_size, self.grid_size])

    @staticmethod
    def build(grid_world: GridWorld, seed: int, use_heldout_tasks: bool):
        return Env(
            absorbing_state=grid_world.absorbing_state,
            dense_reward=grid_world.dense_reward,
            episode_length=grid_world.episode_length,
            grid_size=grid_world.grid_size,
            heldout_goals=grid_world.heldout_goals,
            seed=seed,
            terminate_after_goal=grid_world.terminate_after_goal,
            use_heldout_goals=use_heldout_tasks,
        )

    def get_task(self):
        [goal] = self.goals
        return goal

    def reset(self):
        self.current_state = self.reset_fn()
        [s] = self.current_state
        self.t = 0
        self.t_since_goal = -1
        distance = torch.abs(self.get_task() - s).sum()
        tail_length = (
            self.terminate_after_goal
            if (self.terminate_after_goal >= 0)
            else (self.episode_length - distance - 1)
        )
        if self.dense_reward:
            descending = list(range(-distance, 1))
            self.optimal = descending + [0.0] * tail_length
        else:
            self.optimal = (
                [0.0] * distance
                + [1.0]
                + [0.0 if self.absorbing_state else 1.0] * tail_length
            )
        return s.numpy()

    def step(self, action: Union[torch.Tensor, int]):
        if isinstance(action, int):
            action = torch.tensor([action])
        action = action.reshape(1)
        t = torch.tensor([self.t])
        (self.current_state, [self.t_since_goal]), [r], d, i = self.step_fn(
            self.current_state, action, t, torch.tensor([self.t_since_goal])
        )
        [s] = self.current_state
        self.t += 1
        if d:
            i.update(optimal=self.optimal)
        return s.numpy(), r, d, i
