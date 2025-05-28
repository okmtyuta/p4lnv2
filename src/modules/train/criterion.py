from typing import TypedDict
import torch
from scipy import stats
import numpy.typing as npt
import numpy as np

from src.modules.train.types import Criteria




class Criterion:
    def __init__(self):
        self._mse_loss = torch.nn.MSELoss()
        self._l1_loss = torch.nn.L1Loss()

    def root_mean_squared_error(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self.mean_squared_error(output=output, label=label).sqrt()
        return loss

    def mean_squared_error(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._mse_loss(output, label)
        return loss

    def mean_absolute_error(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._l1_loss(output, label)
        return loss

    def pearsonr(self, output: torch.Tensor, label: torch.Tensor) -> np.float32:
        correlation = stats.pearsonr(output.detach(), label.detach()).correlation
        return correlation

    def __call__(self, output: torch.Tensor, label: torch.Tensor) -> Criteria:
        root_mean_squared_error = self.root_mean_squared_error(output=output, label=label)
        mean_squared_error = self.mean_squared_error(output=output, label=label)
        mean_absolute_error = self.mean_absolute_error(output=output, label=label)
        pearsonr = self.pearsonr(output=output, label=label)

        return {
            "root_mean_squared_error": root_mean_squared_error.item(),
            "mean_squared_error": mean_squared_error.item(),
            "mean_absolute_error": mean_absolute_error.item(),
            "pearsonr": pearsonr.item(),
        }

    # def delta(self, output: torch.Tensor, label: torch.Tensor, alpha: float = 0.95):
    #     output_mean = torch.mean(output)
    #     label_mean = torch.mean(label)
    #     output_var = torch.var(output)
    #     label_var = torch.var(label)

    #     diff = output_mean - label_mean
    #     upper = diff + 1.96 * torch.sqrt((output_var / len(output)) + (label_var / len(label)))
    #     lower = diff - 1.96 * torch.sqrt((output_var / len(output)) + (label_var / len(label)))

    #     print(lower, upper)
