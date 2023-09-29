from collections import defaultdict
from dataclasses import asdict, astuple, replace

import pandas as pd
import torch
import torch.nn.functional as F

import data.base
from data.base import Step
from encoder import OffsetEncoder
from envs.grid_world_env import Env
from envs.value_iteration import ValueIteration
from plot import plot_eval_metrics
from pretty import console, render_eval_metrics


def expand_as(x: torch.Tensor, y: torch.Tensor):
    while x.dim() < y.dim():
        x = x[..., None]
    return x.expand_as(y)


class Data(data.base.Data):
    def __init__(
        self,
        alpha: float,
        dense_reward: bool,
        grid_size: int,
        grid_world_args: dict,
        heldout_tasks: list[tuple[int, int]],
        include_goal: bool,
        mask_nonactions: bool,
        n_data: int,
        n_episodes: int,
        steps_per_context: int,
        value_iteration_args: dict,
        yield_every: int,
    ):
        self.dense_reward = dense_reward
        self.grid_size = grid_size
        self.heldout_goals = heldout_tasks
        self.n_data = n_data
        self.n_episodes = n_episodes
        self.steps_per_context = steps_per_context
        self.n_rounds = round((2 * grid_size - 1) / alpha)
        self.yield_every = yield_every

        grid_world = ValueIteration(
            alpha=alpha,
            dense_reward=dense_reward,
            episode_length=self.episode_length,
            **grid_world_args,
            grid_size=grid_size,
            heldout_goals=heldout_tasks,
            n_tasks=n_data,
            use_heldout_goals=False,
        )
        self._include_goal = include_goal

        data = list(self.collect_data(grid_world, **value_iteration_args))
        data = [[*astuple(components), done] for components, done in data]
        components = zip(*data)
        components = [torch.cat(c, dim=1) for c in components]
        *components, done = components
        components = Step(*components)
        if not include_goal:
            components = replace(components, tasks=torch.zeros_like(components.tasks))

        def make_mask(component: torch.Tensor):
            return expand_as(~done, component).roll(dims=[1], shifts=[1])

        masks = Step(
            tasks=make_mask(components.tasks),
            observations=make_mask(components.observations),
            actions=torch.ones_like(components.actions),
            rewards=torch.ones_like(components.rewards),
        )
        if mask_nonactions:
            masks = replace(
                masks,
                tasks=torch.zeros_like(masks.tasks),
                observations=torch.zeros_like(masks.observations),
                rewards=torch.zeros_like(masks.rewards),
            )

        self.tasks = components.tasks
        self.observations = components.observations
        self.actions = components.actions
        self.rewards = components.rewards

        mask = self.cat_sequence(masks)
        data = self.cat_sequence(components)
        sequence = self.split_sequence(data)
        for name, component in asdict(sequence).items():
            assert (getattr(components, name) == component).all()

        data = data.reshape(n_data, -1, self.step_dim)
        mask = mask.reshape(n_data, -1, self.step_dim)
        _, self.steps_per_row, _ = data.shape
        self.unpadded_data = data
        assert [*self.unpadded_data.shape] == [
            n_data,
            self.steps_per_row,
            self.step_dim,
        ]
        pad_value = self.unpadded_data.max().item() + 1
        self.data = F.pad(data, (0, 0, steps_per_context, 0), value=pad_value)
        self.mask = F.pad(mask, (0, 0, steps_per_context, 0), value=0)
        self.min_value = self.data.min().item()

    def __getitem__(self, idx):
        i, j = self.index_1d_to_2d(idx)
        jj = slice(j, j + self.steps_per_context)
        return self.data[i, jj].view(-1), self.mask[i, jj].view(-1)

    def __len__(self):
        return self.n_data * self.steps_per_row

    @property
    def context_size(self):
        return self.steps_per_context * self.step_dim

    @property
    def dims(self):
        _, _, goal_dim = self.tasks.shape
        _, _, obs_dim = self.observations.shape
        return Step(tasks=goal_dim, observations=obs_dim, actions=1, rewards=1)

    @property
    def encoder(self):
        return OffsetEncoder(self.min_value)

    @property
    def episode_length(self):
        return 1 + self.grid_size**2

    @property
    def episodes_per_rollout(self):
        return self.n_rounds * self.n_episodes

    @property
    def include_goal(self):
        return self._include_goal

    @property
    def max_regret(self):
        if self.dense_reward:
            return sum(range(self.episode_length))
        else:
            return 1

    def build_env(self, seed: int, use_heldout_tasks: bool):
        return Env(
            dense_reward=self.dense_reward,
            episode_length=self.episode_length,
            grid_size=self.grid_size,
            heldout_goals=self.heldout_goals,
            seed=seed,
            use_heldout_goals=use_heldout_tasks,
        )

    def cat_sequence(self, step: Step):
        step = replace(step, rewards=step.rewards[..., None])
        data = torch.cat(
            astuple(step),
            dim=-1,
        )
        n_data, *_ = data.shape
        return data.long().view(n_data, -1)

    def collect_data(self, grid_world: ValueIteration, **kwargs):
        console.log("Value iteration...")
        for t, (V, Pi) in enumerate(
            (grid_world.value_iteration(**kwargs, n_rounds=self.n_rounds))
        ):
            g, s, a, r, d = grid_world.get_trajectories(
                Pi=Pi, n_episodes=self.n_episodes
            )
            console.log(
                f"Round: {t}. Reward: {r.sum(-1).mean().item():.2f}. Value: {V.mean().item():.2f}."
            )
            if t % self.yield_every == 0:
                yield Step(tasks=g, observations=s, actions=a, rewards=r), d

    @property
    def eval_metric_name(self) -> str:
        return "regret"

    def index_1d_to_2d(self, index):
        row = index // self.steps_per_row
        col = index % self.steps_per_row
        return (row, col)

    def split_sequence(self, sequence: torch.Tensor):
        n_batch, _ = sequence.shape
        sequence = sequence.reshape(n_batch, -1, self.step_dim)
        dims = astuple(self.dims)
        components = sequence.split(dims, dim=-1)
        components = Step(*components)
        components.rewards.squeeze_(-1)
        return components

    def get_metrics(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        sequence: torch.Tensor,
    ):
        n_batch, seq_len = sequence.shape
        n_batch2, seq_len2, _ = logits.shape
        assert n_batch == n_batch2
        assert seq_len == seq_len2 + 1

        prefix = sequence[:, :1]
        preds = logits.argmax(-1)
        preds = self.encoder.decode(preds)
        preds = torch.cat([prefix, preds], dim=1)
        tgts = sequence
        split_preds = self.split_sequence(preds)
        split_tgts = self.split_sequence(tgts)
        split_masks = self.split_sequence(mask)

        acc = {}
        acc["(total) accuracy"] = (preds == tgts)[mask.bool()]
        iterator = list(
            zip(
                asdict(split_preds).items(),
                asdict(split_tgts).items(),
                asdict(split_masks).items(),
            )
        )
        for (name, pred), (name2, tgt), (name3, mask) in iterator:
            assert name == name2 == name3
            total_accuracy = (pred == tgt)[mask.bool()]
            acc[f"(total) {name} accuracy"] = total_accuracy

        table = defaultdict(list)
        for i in range(seq_len):
            for (name, pred), (name2, tgt), (name3, mask) in iterator:
                assert name == name2 == name3
                _, component_seq_len, *_ = pred.shape

                if i >= component_seq_len:
                    continue

                def get_chunk(x):
                    if x.ndim == 2:
                        x = x[..., None]
                    return x[:, i : i + 1]

                pred_chunk = get_chunk(pred)
                tgt_chunk = get_chunk(tgt)
                mask_chunk = get_chunk(mask)
                if mask_chunk.sum() > 0:
                    accuracy = (pred_chunk == tgt_chunk)[mask_chunk.bool()]
                    table[f"{name} accuracy"].append(accuracy.float().mean().item())

        log = {k: v.float().mean().item() for k, v in acc.items()}
        return log, table

    def plot_eval_metrics(self, df: pd.DataFrame) -> list[str]:
        return plot_eval_metrics(
            df, name=self.eval_metric_name, ymin=0, ymax=self.max_regret
        )

    def render_eval_metrics(self, *metric: float) -> list[str]:
        return render_eval_metrics(*metric, max_num=self.max_regret)
