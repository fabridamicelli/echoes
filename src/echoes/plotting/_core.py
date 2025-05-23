"""
Plotting functions often needed.
Not extremely well polished, rather a tool for quick visualization.
"""

from __future__ import annotations  # TODO: Remove after dropping python 3.9

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from echoes.esn import ESNGenerator, ESNRegressor


def set_mystyle():
    """Set context and a couple of defaults for nicer plots."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.4,
        rc={"grid.linestyle": "--", "grid.linewidth": 0.8},
    )


def plot_predicted_ts(
    ts_true: np.ndarray | list | pd.Series,
    ts_pred: np.ndarray | list | pd.Series,
    start: int | None = None,
    end: int | None = None,
    ax: plt.Axes | None = None,
    title: str = "",
    figsize: tuple = (6, 2),
    legend: bool = True,
):
    """
    Arguments:
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

    Returns:
        ax: matplotlib Axes
            Returns the Axes object with the plot drawn onto it.
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
    return ax


def plot_reservoir_activity(
    esn: ESNRegressor | ESNGenerator,
    neurons: np.ndarray | list,
    train: bool = False,
    pred: bool = True,
    start: int | None = None,
    end: int | None = None,
    figsize: tuple = (15, 9),
    **kwargs,
):
    """
    Plot the activity, ie time series of states, of the reservoir
    neurons.

    Arguments:
        esn: ESNPredictive, ESNGenerative
            Instances of ESN after fitting and/or prediction.
        neurons: np.ndarray or List
            List of reservoir neurons indices whose time series will be plotted.
        train: bool, optional
            If True, the time series during training will be plotted.
            Either train or pred must be True, but only one of the two.
        pred: bool, optional
            If True, the time series during prediction will be plotted.
            Either train or pred must be True, but only one of the two.
        start: int, optional
            Plot will be timeseries[start: end].
        end: int, optional
            Plot will be timeseries[start: end].
        suptitle: str, optional
            Plot suptitle.
        figsize: tuple
            Figure size.
            Default (15, 10).
        kwargs: dict
            Plotting kwargs passed to plt.plot

    Returns:
        fig: plt.figure
            Figure object for fine tuning.
    """
    assert train or pred, "either train or pred must be True"
    assert not (train and pred), "only one of train or pred can be True"

    n_neurons = len(neurons)
    # Grab time series to plot
    ts = esn.states_pred_ if pred else esn.states_train_

    # Plot test
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_neurons / 3)), ncols=3, figsize=figsize
    )

    if "linewidth" in kwargs:
        linewidth = kwargs.pop("linewidht")
    else:
        linewidth = 3
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = ".6"

    for neuron_idx, neuron in enumerate(neurons):
        ax = axes.flat[neuron_idx]
        ax.plot(ts[start:end, neuron], linewidth=linewidth, color=color, **kwargs)
        ax.set_ylabel("state")
        ax.set_xlabel("time")
        ax.set_title(f"reservoir neuron idx: {neuron}")

    # Delete unnecessary axes
    if n_neurons % 3 == 1:
        fig.delaxes(axes.flat[-1])
        fig.delaxes(axes.flat[-2])
    elif n_neurons % 3 == 2:
        fig.delaxes(axes.flat[-1])

    fig.tight_layout()
    return fig
