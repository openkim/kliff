from typing import Union

import numpy as np
import torch


def MSE_loss(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    weights: Union[np.ndarray, torch.Tensor] = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Compute the mean squared error (MSE) of the residuals, with the option to
    weight the residuals.

    Args:
        predictions: The predicted values.
        targets: The target values.
        weights: The weights to apply to the residuals. Default is 1.0.

    Returns:
        The MSE of the residuals.
    """
    residuals = predictions - targets
    if isinstance(residuals, (np.ndarray, float, np.float64)):
        return np.mean((residuals * weights) ** 2)
    else:
        return torch.mean(
            (residuals**2)
            * torch.asarray(weights, dtype=residuals.dtype, device=residuals.device)
        )


def MAE_loss(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    weights: Union[np.ndarray, torch.Tensor] = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Compute the mean absolute error (MAE) of the residuals, with the option to
    weight the residuals.

    Args:
        predictions: The predicted values.
        targets: The target values.
        weights: The weights to apply to the residuals. Default is 1.0.

    Returns:
        The MAE of the residuals.
    """
    residuals = predictions - targets
    if isinstance(residuals, (np.ndarray, float, np.float64)):
        return np.mean(np.abs(residuals * weights))
    else:
        return torch.mean(
            torch.abs(residuals)
            * torch.asarray(weights, dtype=residuals.dtype, device=residuals.device)
        )
