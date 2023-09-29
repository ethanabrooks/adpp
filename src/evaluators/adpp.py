from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import evaluators.ad
from evaluators.ad import StepResult, clamp

# from plot import plot_rollout


@dataclass
class Evaluator(evaluators.ad.Evaluator):
    n_actions: int

    def make_rollout(self, *args, **kwargs):
        return Rollout(*args, **kwargs, n_actions=self.n_actions)


@dataclass
class PlanningRollout(evaluators.ad.Rollout):
    history: torch.Tensor
    t: int
    episode_t: np.ndarray

    def __post_init__(self):
        pass

    @property
    def episode_length(self):
        return 1 + self.dataset.episode_length - self.episode_t.max()

    @property
    def episodes_per_rollout(self):
        return 1

    def index(self, t: int):
        return super().index(t + self.t)

    def init_history(self):
        T = self.dataset.episode_length * self.dataset.episodes_per_rollout
        required_width = self.index(T + 1)
        _, L = self.history.shape
        pad = required_width - L
        if pad > 0:
            return F.pad(self.history, (0, pad), value=self.dataset.pad_value)
        else:
            return self.history

    def reset(self, *_, **__):
        pass  # No reset. Only one episode.

    def step(self, action: torch.Tensor, history: torch.Tensor, t: int) -> torch.Tensor:
        O = self.dataset.dims.observations

        [reward] = self.predict_many(history, t, *self.idxs.rewards).T
        assert [*reward.shape] == [self.n_rollouts]

        i, j = self.index(t) + self.idxs.tasks
        history[:, i:j] = self.task

        observation = self.predict_many(history, t + 1, *self.idxs.observations)
        assert [*observation.shape] == [self.n_rollouts, O]

        done = t + 1 == self.episode_length
        done = np.full(reward.shape, fill_value=done, dtype=bool)

        return StepResult(
            reward=reward.cpu().numpy(),
            observation=observation.cpu().numpy(),
            done=done,
            info=[{} for _ in range(self.n_rollouts)],
        )


@dataclass
class Rollout(evaluators.ad.Rollout):
    n_actions: int

    def get_action(self, history: torch.Tensor, t: int, episode_t) -> torch.Tensor:
        N = self.n_actions
        i, *_, j = (history[0] != self.dataset.pad_value).nonzero()

        # for each row in history, we perform N rollouts
        repeated_history = history.repeat(N, 1)  # repeat history N times
        planner = PlanningRollout(
            dataset=self.dataset,
            envs=self.envs,
            episode_t=episode_t,
            gamma=self.gamma,
            history=repeated_history,
            n_rollouts=self.n_rollouts * N,
            net=self.net,
            t=t,
            task=self.task.repeat(N, 1),
        )
        rollouts = list(planner.rollout())  # rollout for each row in repeated_history
        rollouts = pd.DataFrame.from_records(rollouts)
        rollouts["row"] = rollouts.n % self.n_rollouts  # which row in original history
        group = rollouts.groupby("row")
        # group.apply(self.check_identical_elements, i, j)
        self.check_identical_elements  # whitelist
        idx = group.metric.idxmax()  # index of best rollout per original row

        # extract actions
        histories = rollouts.loc[idx].history.tolist()
        histories = torch.stack(histories).cuda()
        i, j = self.index(t) + self.idxs.actions
        actions = histories[:, i:j]
        return clamp(actions, self.envs.action_space)

    def check_identical_elements(self, group: pd.DataFrame, i: int, j: int):
        histories = group["history"].tolist()
        # Convert to tensor for easier manipulation
        histories_tensor = torch.stack(histories)

        # Find the index where elements stop being pad_value for the first time
        non_pad_column = histories_tensor[:, i:j]
        first_row = non_pad_column[0]
        assert torch.all(first_row[None] == non_pad_column)
