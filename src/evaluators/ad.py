from dataclasses import dataclass

import numpy as np
import torch
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


@dataclass(frozen=True)
class Step:
    action: np.ndarray
    reward: np.ndarray
    observation: np.ndarray
    done: np.ndarray
    info: list[dict]


@dataclass
class Rollout:
    dataset: Data
    envs: SubprocVecEnv
    gamma: float
    n_rollouts: int
    net: GPT

    def __post_init__(self):
        N = self.n_rollouts
        O = get_dim(self.envs.observation_space)

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

    def get_action(self, ctx: torch.Tensor) -> torch.Tensor:
        A = get_dim(self.envs.action_space)
        return torch.stack(list(self.predict_many(ctx, start=-A - 1, end=-1)), dim=-1)

    def predict(self, ctx: torch.Tensor, i: int) -> torch.Tensor:
        dataset = self.dataset
        net = self.net
        N = self.n_rollouts
        K = dataset.n_tokens + 1

        # pass through net
        logits: torch.Tensor
        logits, _ = net.forward(ctx)
        assert [*logits.shape] == [N, net.context_size, 1 + dataset.n_tokens]

        # sample action
        probs = logits[:, i].softmax(dim=-1)

        assert [*probs.shape] == [N, K]
        [prediction] = torch.multinomial(probs, num_samples=1).T
        return prediction

    def predict_many(self, ctx: torch.Tensor, start: int, end: int):
        for i in range(start, end):
            prediction = self.predict(ctx, i)
            yield prediction
            ctx[:, i] = prediction

    def rollout(self):
        A = get_dim(self.envs.action_space)
        N = self.n_rollouts
        O = get_dim(self.envs.observation_space)
        S = self.dataset.step_dim
        T = self.dataset.episode_length * self.dataset.episodes_per_rollout
        W = self.net.context_size
        envs = self.envs
        dataset = self.dataset
        observation = self.first_observation

        # dummies
        dummy_action = torch.tensor(dataset.pad_value).repeat(N, A).cuda()
        dummy_reward = torch.tensor(dataset.pad_value).repeat(N).cuda()

        T = dataset.episode_length * dataset.episodes_per_rollout
        episode_count = np.zeros(N, dtype=int)
        episode_rewards = np.zeros((N, dataset.episode_length))
        episode_t = np.zeros(N, dtype=int)
        ctx = torch.full(
            (N, W + T * S),
            dataset.pad_value,
            dtype=torch.long,
            device="cuda",
        )

        for t in tqdm(range(T)):
            # create sequence
            observation = torch.tensor(observation).cuda()

            ## create context and pad
            end = t + W + 1
            start = end - S
            ctx[:, start:end] = self.dataset.cat_sequence(
                self.task, observation, dummy_action, dummy_reward
            )

            step = self.step(ctx[:, t : t + W + 1])

            assert [*step.observation.shape] == [N, O]
            assert [*step.reward.shape] == [N]
            assert [*step.done.shape] == [N]
            assert len(step.info) == N

            ctx[:, start:end] = self.dataset.cat_sequence(
                self.task,
                observation,
                step.action.clone().detach().cuda(),
                torch.tensor(step.reward).cuda(),
            )
            observation = step.observation
            episode_rewards[np.arange(N), episode_t] = step.reward
            episode_t += 1

            for n, (d, ec, er, et, i) in enumerate(
                zip(step.done, episode_count, episode_rewards, episode_t, step.info)
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

    def step(self, ctx: torch.Tensor):
        action = self.get_action(ctx)
        action = clamp(action, self.envs.action_space)
        observation, reward, done, info = self.envs.step(
            action.squeeze(0).cpu().numpy()
        )
        return Step(
            action=action, reward=reward, observation=observation, done=done, info=info
        )
