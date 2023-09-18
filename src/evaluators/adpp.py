from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

import evaluators.ad
from evaluators.ad import Step, clamp, get_dim


class Evaluator(evaluators.ad.Evaluator):
    @staticmethod
    def make_rollout(*args, **kwargs):
        return Rollout(*args, **kwargs)


@dataclass
class PlanningRollout(evaluators.ad.Rollout):
    first_observation: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def step(self, ctx: torch.Tensor) -> torch.Tensor:
        A = get_dim(self.envs.action_space)
        O = get_dim(self.envs.observation_space)
        S = self.dataset.step_dim
        action = self.get_action(ctx)
        ctx[:, -A - 1 : -1] = action
        assert [*action.shape] == [self.n_rollouts, A]
        action = clamp(action, self.envs.action_space)
        _, w = ctx.shape
        [reward] = self.predict(ctx, w - 2, w - 1).T
        assert [*reward.shape] == [self.n_rollouts]
        ctx[:, -1] = reward
        ctx = ctx[:, S:]  # drop first step
        ctx = F.pad(ctx, (0, S), value=self.dataset.pad_value)  # pad out last step
        observation = self.predict(ctx, -S, -S + O)
        assert [*observation.shape] == [self.n_rollouts, O]
        done = torch.zeros_like(reward, dtype=torch.bool)
        return Step(
            action=action, reward=reward, observation=observation, done=done, info=[]
        )


class Rollout(evaluators.ad.Rollout):
    def get_action(self, ctx: torch.Tensor) -> torch.Tensor:
        PlanningRollout(**asdict(self))
        raise NotImplementedError
