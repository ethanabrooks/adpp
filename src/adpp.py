import torch

import evaluators.ad


class Evaluator(evaluators.ad.Evaluator):
    def get_action(self, ctx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
