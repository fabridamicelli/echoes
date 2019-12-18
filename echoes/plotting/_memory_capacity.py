"""
Plotting functions related to the Memory Capacity task.
"""
from typing import List, Union, Any

import numpy as np
import matplotlib.pyplot as plt

from ._core import plot_predicted_ts


def plot_forgetting_curve(
    lags: Union[List, np.ndarray],
    forgetting_curve: np.ndarray,
    ax: plt.Axes = None,
    **kwargs: Any,
) -> None:
    """
    Plot forgetting curve, ie, memory capacity (MC) vs lag.

    Parameters
    ----------
    lags: np.ndarray or List
        Sequence of lags used in the memory capacity task.
    forgetting_curve: np.ndarray
        Sequence of results from the memory task.
    ax: plt.Axes, optional
        If given plot will use this axes.
    kwargs: mapping, optional
        Plotting args passed to ax.plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(lags, forgetting_curve, **kwargs)
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$MC_k$")


def plot_mc_predicted_ts(
    lags: Union[List, np.ndarray],
    outputs_true: np.ndarray,
    outputs_pred: np.ndarray,
    start: int = None,
    end: int = None,
) -> None:
    """
    Plot true and predicted time series coming from memory capacity
    task for all lags.

    Parameters
    ----------
    lags: np.ndarray or List
        Delays to be evaluated (memory capacity).
        For example: np.arange(1, 31, 5).
    ouputs_true: np.ndarray of shape (len(ts), len(n_lags))
        Target time series used for testing the model.
    ouputs_pred: np.ndarray of shape (len(ts), len(n_lags))
        Predicted time series resulting from testing the model.
    start/end: int, optional
        Plot will we timeseries[start: end], to exclude transient.
    """
    assert (
        outputs_true.shape == outputs_pred.shape
    ), "true and pred outputs must have same shape"
    assert (
        len(lags) == outputs_true.shape[1]
    ), "second dimension of outputs must equal len(lags)"

    n_lags = len(lags)

    # Plot test
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_lags / 2)), ncols=2, figsize=(18, 2.0 * n_lags)
    )
    for lag_idx, lag in enumerate(lags):
        ax = axes.flat[lag_idx]

        plot_predicted_ts(
            outputs_true[:, lag_idx],
            outputs_pred[:, lag_idx],
            start=start,
            end=end,
            title=f"lag = {lag}",
            ax=ax,
            legend=False,
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=20,
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    if n_lags % 2 != 0:
        fig.delaxes(axes.flat[-1])
    fig.tight_layout()
