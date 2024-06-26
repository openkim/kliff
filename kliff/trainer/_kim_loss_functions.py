import numpy as np


def MSE_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    r"""
    Compute the mean squared error (MSE) of the residuals.

    Args:

    Returns:
        The MSE of the residuals.
    """
    residuals = predictions - targets
    return np.mean(residuals**2)
