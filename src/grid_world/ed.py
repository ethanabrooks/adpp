import grid_world.ad
from grid_world.value_iteration import ValueIteration
from pretty import console


class Data(grid_world.ad.Data):
    def collect_data(self, grid_world: ValueIteration, **kwargs):
        console.log("Value iteration...")
        for t, (V, Pi) in enumerate(
            (grid_world.value_iteration(**kwargs, n_rounds=self.n_rounds))
        ):
            step, dones = grid_world.get_trajectories(Pi=Pi, n_episodes=self.n_episodes)
            console.log(
                f"Round: {t}. Reward: {step.rewards.sum(-1).mean().item():.2f}. Value: {V.mean().item():.2f}."
            )
        yield step, dones

    @property
    def episodes_per_rollout(self):
        return self.n_episodes
