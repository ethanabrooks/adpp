import torch

import data.base
from tabular.grid_world import GridWorld


def expand_as(x: torch.Tensor, y: torch.Tensor):
    while x.dim() < y.dim():
        x = x[..., None]
    return x.expand_as(y)


class Data(data.base.Data):
    def __init__(
        self,
        grid_size: int,
        n_data: int,
    ):
        episode_length = 1 + grid_size * 2
        grid_world = GridWorld(grid_size, n_data)
        (self.goals, self.observations, self.actions, self.rewards, self.done,) = (
            *components,
            _,
        ) = grid_world.get_trajectories(
            episode_length=episode_length,
            n_episodes=1,
            Pi=grid_world.compute_policy_towards_goal(),
        )
        masks = [
            expand_as(~self.done, c).roll(dims=[1], shifts=[1])
            for c in [self.goals, self.observations]
        ] + [torch.ones_like(c) for c in [self.actions, self.rewards]]
        self.mask = self.cat_sequence(*masks)
        self.data = self.cat_sequence(*components)
        sequence = self.split_sequence(self.data)
        for name, component in dict(
            observations=self.observations, actions=self.actions, rewards=self.rewards
        ).items():
            assert (sequence[name] == component).all()
        self.data = self.data.cuda()
        self.mask = self.mask.cuda()

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

    def __len__(self):
        return len(self.data)

    @property
    def _dims(self):
        _, _, goal_dim = self.goals.shape
        _, _, obs_dim = self.observations.shape
        return [goal_dim, obs_dim, 1, 1]

    def cat_sequence(self, goals, observations, actions, rewards):
        data = torch.cat(
            [goals, observations, actions[..., None], rewards[..., None]],
            dim=-1,
        )
        n_data, _, _ = data.shape
        return data.long().reshape(n_data, -1).contiguous()

    def split_sequence(self, sequence: torch.Tensor):
        n_batch, _ = sequence.shape
        sequence = sequence.reshape(n_batch, -1, self.step_dim)
        goals, observations, actions, rewards = sequence.split(self._dims, dim=-1)
        actions = actions.squeeze(-1)
        rewards = rewards.squeeze(-1)
        return dict(
            goals=goals, observations=observations, actions=actions, rewards=rewards
        )

    def get_metrics(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        sequence: torch.Tensor,
        steps_per_graph: int,
    ):
        n_batch, seq_len = sequence.shape
        n_batch2, seq_len2, _ = logits.shape
        assert n_batch == n_batch2
        assert seq_len == seq_len2 + 1

        prefix = sequence[:, :1]
        preds = torch.cat([prefix, logits.argmax(-1)], dim=1)
        tgts = sequence
        split_preds = self.split_sequence(preds)
        split_tgts = self.split_sequence(tgts)
        split_masks = self.split_sequence(mask)

        acc = {}
        acc["(total) accuracy"] = (preds == tgts)[mask.bool()]
        iterator = list(
            zip(
                split_preds.items(),
                split_tgts.items(),
                split_masks.items(),
            )
        )
        for (name, pred), (name2, tgt), (name3, mask) in iterator:
            assert name == name2 == name3
            total_accuracy = (pred == tgt)[mask.bool()]
            acc[f"(total) {name} accuracy"] = total_accuracy

        for i in range(seq_len):
            for (name, pred), (name2, tgt), (name3, mask) in iterator:
                assert name == name2 == name3
                _, component_seq_len, *_ = pred[mask].shape
                graphs_per_component = component_seq_len // steps_per_graph

                if i >= graphs_per_component:
                    continue

                def get_chunk(x, start, end):
                    if x.ndim == 2:
                        x = x[..., None]
                    return x[:, start:end]

                start = i * steps_per_graph
                end = (i + 1) * steps_per_graph

                pred_chunk = get_chunk(pred, start, end)
                tgt_chunk = get_chunk(tgt, start, end)
                mask_chunk = get_chunk(mask, start, end)
                if mask_chunk.sum() > 0:
                    acc[f"({i}) {name} accuracy"] = (pred_chunk == tgt_chunk)[
                        mask_chunk.bool()
                    ]

        return {k: v.float().mean().item() for k, v in acc.items()}
