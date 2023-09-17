import torch

import evaluators.ad


class Evaluator(evaluators.ad.Evaluator):
    def predict(self, ctx: torch.Tensor, indices: list[int]) -> torch.Tensor:
        raise NotImplementedError
