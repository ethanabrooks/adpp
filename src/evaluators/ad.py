from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Discrete, MultiDiscrete, Space
from tqdm import tqdm

from data.base import Data
from envs.parallel.dummy_vec_env import DummyVecEnv
from envs.parallel.subproc_vec_env import SubprocVecEnv
from models import GPT


def get_return(*rewards: float, gamma: float) -> float:
    actual_return = 0
    for r in rewards[::-1]:
        actual_return = r + gamma * actual_return
    return actual_return


def get_dim(space: Space) -> int:
    if isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiDiscrete):
        return space.nvec.size
    else:
        raise NotImplementedError


def clamp(action: torch.Tensor, space: Space):
    if isinstance(space, Discrete):
        return torch.clamp(action, min=0, max=space.n - 1)
    elif isinstance(space, MultiDiscrete):
        return torch.clamp(action, min=0, max=space.nvec - 1)
    else:
        raise NotImplementedError


def get_metric(
    info,
    rewards,
    gamma,
):
    actual_return = get_return(*rewards, gamma=gamma)
    optimal = info.get("optimal", None)
    if optimal is None:
        return "return", actual_return
    else:
        optimal_return = get_return(*optimal, gamma=gamma)
        regret = optimal_return - actual_return
        return "regret", regret


class Evaluator:
    @classmethod
    def evaluate(cls, dataset: Data, dummy_vec_env: bool, n_rollouts: int, **kwargs):
        N = n_rollouts
        env_fns = [dataset.build_env for _ in range(N)]
        envs: SubprocVecEnv
        envs = DummyVecEnv(env_fns) if dummy_vec_env else SubprocVecEnv(env_fns)
        try:
            evaluator = cls.make_rollout(
                dataset=dataset, envs=envs, n_rollouts=n_rollouts, **kwargs
            )
            yield from evaluator.rollout()
        finally:
            envs.close()

    @staticmethod
    def make_rollout(*args, **kwargs):
        return Rollout(*args, **kwargs)


@dataclass
class Rollout:
    dataset: Data
    envs: SubprocVecEnv
    gamma: float
    n_rollouts: int
    net: GPT

    def __post_init__(self):
        N = self.n_rollouts
        T = self.dataset.episode_length * self.dataset.episodes_per_rollout
        O = get_dim(self.envs.observation_space)
        A = get_dim(self.envs.action_space)

        # task
        task = torch.tensor(self.envs.get_task()).cuda()
        assert [*task.shape] == [N, 2]
        if not self.dataset.include_goal:
            task = torch.zeros_like(task)
        task_dim = get_dim(self.envs.task_space)
        assert [*task.shape] == [N, task_dim]
        self.task = task

        # observation
        observation = self.envs.reset()
        O = get_dim(self.envs.observation_space)
        assert [*observation.shape] == [N, O]
        self.first_observation = observation

        task_dim = get_dim(self.envs.task_space)
        self.tasks = torch.zeros(N, T, task_dim).cuda()
        self.observations = torch.zeros(N, T, O).cuda()
        self.actions = torch.zeros(N, T, A).cuda()
        self.rewards = torch.zeros(N, T).cuda()

    def get_action(self, ctx: torch.Tensor) -> torch.Tensor:
        dataset = self.dataset
        net = self.net
        N = self.n_rollouts
        A = get_dim(self.envs.action_space)

        # pass through net
        logits: torch.Tensor
        logits, _ = net.forward(ctx)
        assert [*logits.shape] == [N, dataset.context_size - 1, 1 + dataset.n_tokens]

        # access action logits
        logits = logits[:, :-1]  # exclude reward prediction
        logits = logits[:, -A:]  # only consider last action

        # sample action
        probs = logits.softmax(dim=-1)
        assert [*probs.shape] == [N, 1, dataset.n_tokens + 1]
        return torch.multinomial(probs.squeeze(1), num_samples=1)

    def rollout(self):
        N = self.n_rollouts
        O = get_dim(self.envs.observation_space)
        A = get_dim(self.envs.action_space)
        envs = self.envs
        dataset = self.dataset
        observation = self.first_observation

        # actions
        dummy_action = torch.tensor(dataset.pad_value).repeat(N, A).cuda()

        # reward
        dummy_reward = torch.tensor(dataset.pad_value).repeat(N, 1).cuda()

        T = dataset.episode_length * dataset.episodes_per_rollout
        tasks = self.tasks
        observations = self.observations
        actions = self.actions
        rewards = self.rewards
        episode_count = np.zeros(N, dtype=int)
        episode_rewards = np.zeros((N, dataset.episode_length))
        episode_t = np.zeros(N, dtype=int)

        for t in tqdm(range(T)):
            tasks[:, t] = self.task
            observations[:, t] = torch.tensor(observation).cuda()

            # create sequence
            sequence = [
                tasks[:, : t + 1],
                observations[:, : t + 1],
                torch.cat([actions[:, :t], dummy_action[:, None]], dim=1),
                torch.cat([rewards[:, :t], dummy_reward], dim=1),
            ]

            ## create context and pad
            ctx = dataset.cat_sequence(*sequence)
            assert [*ctx.shape] == [N, (t + 1) * dataset.step_dim]
            pad_size = dataset.context_size - ctx.numel() // N
            ctx = F.pad(ctx, (pad_size, 0), value=dataset.pad_value)
            action = self.get_action(ctx)
            action = clamp(action, self.envs.action_space)
            observation, reward, done, info = envs.step(action.squeeze(0).cpu().numpy())
            assert [*observation.shape] == [N, O]
            assert [*reward.shape] == [N]
            assert [*done.shape] == [N]
            assert len(info) == N
            actions[:, t] = action
            rewards[:, t] = torch.tensor(reward).cuda()
            episode_rewards[np.arange(N), episode_t] = reward
            episode_t += 1

            for n, (d, ec, er, et, i) in enumerate(
                zip(done, episode_count, episode_rewards, episode_t, info)
            ):
                assert isinstance(d, (bool, np.bool_))
                assert isinstance(i, dict)
                if d:
                    name, x = get_metric(info=i, rewards=er[:et], gamma=self.gamma)
                    yield dict(n=n, name=name, t=ec, metric=x)
                    episode_count[n] += 1
                    episode_rewards[n] = 0
                    episode_t[n] = 0
                    observation[n] = envs.reset(n)
