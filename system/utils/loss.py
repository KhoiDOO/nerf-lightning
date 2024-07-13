from torch import Tensor

import torch


class MSE:
    def __init__(self, reduction: str) -> None:
        if reduction not in ['mean', 'sum']:
            raise ValueError("Only support reduction in ['mean', 'sum']")
        else:
            self.reduction = reduction
    
    def __call__(self, pred:Tensor, target:Tensor) -> torch.Any:
        error = ((pred - target) ** 2)

        if self.reduction == 'sum':
            return error.sum()
        elif self.reduction == 'mean':
            return error.mean()