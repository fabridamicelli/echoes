"""
Plotting functions often needed.
Not extremely well polished, rather a tool for quick visualization.
"""
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from echoes.esn import ESNGenerative, ESNPredictive


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


def plot_reservoir_activity(
    esn: Union[ESNPredictive, ESNGenerative],
    neurons: Union[np.ndarray, List],
    train: bool = False,
    pred: bool = True,
    start: int = None,
    end: int = None,
    figsize: Tuple = (15, 9)
):
    """
    Plot the activity, ie time series of states, of the reservoir
    neurons.

    esn: EchoStateNetwork
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
    """
    assert train or pred, "either train or pred must be True"
    assert not (train and pred), "only one of train or pred can be True"

    n_neurons = len(neurons)
    # Grab time series to plot
    ts = esn.states_pred_ if pred else esn.states_train_

    # Plot test
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_neurons / 3)), ncols=3, figsize=figsize)

    for neuron_idx, neuron in enumerate(neurons):
        ax = axes.flat[neuron_idx]

        ax.plot(ts[start:end, neuron], color=".52", linewidth=3)
        ax.set_ylabel("state")
        ax.set_xlabel("time")
        ax.set_title(f"reservoir neuron # {neuron}")

    # Delete unnecessary axes
    if n_neurons % 3 == 1:
        fig.delaxes(axes.flat[-1])
        fig.delaxes(axes.flat[-2])
    elif n_neurons % 3 == 2:
        fig.delaxes(axes.flat[-1])

    fig.tight_layout()
