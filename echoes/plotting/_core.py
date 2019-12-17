"""
Plotting functions often needed.
Not extremely well polished but provide with a tool for quick visualization.
"""
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_mystyle():
    """Set context and a couple of defaults for nicer plots."""
    sns.set(
        context="paper",
        style="whitegrid",
        font_scale=1.4,
        font="Helvetica",
        rc={"grid.linestyle": "--", "grid.linewidth": 0.8},
    )


# TODO kwargs for customizing plot,legend, etc.
def plot_predicted_ts(
    ts_true: Union[np.ndarray, List, pd.Series],
    ts_pred: Union[np.ndarray, List, pd.Series],
    start: int = None,
    end: int = None,
    ax: plt.Axes = None,
    title: str = "",
    figsize: Tuple = (6, 2),
    legend: bool = True,
) -> None:
    """
    Parameters
    ----------
    ts_true: np.ndarray, List, pd.Series
        Target time series.
    ts_pred: np.ndarray, List, pd.Series
        Predicted time series.
    start: int, optional
        Plot will be timeseries[start: end].
    end: int, optional
        Plot will be timeseries[start: end].
    ax: plt.Axes, optional
        Axes to plot on. If None, a new figure is created.
        Default None
    title: str,optional
        Plot title.
    figsize: tuple
        Figure size.
        Default (6, 2).
    legend: bool
        If True, legend is added ("target", "predicted").
    """
    if isinstance(ts_true, pd.Series):
        ts_true = ts_true.values
    if isinstance(ts_pred, pd.Series):
        ts_pred = ts_pred.values
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)
    ax.plot(ts_true[start:end], color="steelblue", label="target", linewidth=5.5)
    ax.set_xlabel("time")

    ax.plot(
        ts_pred[start:end],
        linestyle="--",
        color="orange",
        linewidth=2,
        label="prediction",
    )
    ax.set_ylabel("output value")
    ax.set_xlabel("time")

    if legend:
        ax.legend()
