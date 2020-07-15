"""
Numba accelerated implementation reservoir of leaky neurons.
These functions should not be used directly, but via the class ReservoirLeakyNeurons.
"""
from typing import Callable, Optional, Union

import numpy as np
from numba import njit


@njit
def update_state(
    state_t: np.ndarray = None,
    X_t: np.ndarray = None,
    y_t: np.ndarray = None,
    W_in: np.ndarray = None,
    W: np.ndarray = None,
    W_fb: np.ndarray = None,
    bias: Union[np.ndarray, float] = 1,
    feedback: bool = False,
    activation: Callable = None,
    leak_rate: float = 1,
    noise: float = 0,
) -> np.ndarray:
    """
    Return states vector after one time step.
    Notes: Keep this function as autonomous as possible for numba.
    """
    n_reservoir = len(W)
    new_state = W_in @ X_t + W @ state_t + bias

    if feedback:
        new_state += W_fb @ y_t

    new_state = activation(new_state)

    #TODO: check noise: is -0.5 shift necessary?
    if noise:
        new_state += noise * (np.random.rand(n_reservoir) - 0.5)

    # Apply leakage
    if leak_rate < 1:
        new_state = leak_rate * new_state + (1 - leak_rate) * state_t

    return new_state


@njit
def harvest_states(
    X: np.ndarray,
    y: np.ndarray,
    initial_state: Union[np.ndarray, None] = None,
    W_in: np.ndarray = None,
    W: np.ndarray = None,
    W_fb: np.ndarray = None,
    bias: Union[np.ndarray, float] = 1,
    feedback: bool = False,
    activation: Callable = None,
    leak_rate: float = 1,
    noise: float = 0,
) -> np.ndarray:
    """
    Given inputs/outputs X/y, run activity and harvest reservoir states.
    """
    n_reservoir = len(W)
    n_time_steps = X.shape[0]
    states = np.zeros((n_time_steps, n_reservoir))
    if initial_state is not None:
        states[0, :] = initial_state

    for t in range(1, n_time_steps):
        states[t, :] = update_state(
            state_t=states[t - 1],
            X_t=X[t, :],
            y_t=y[t - 1, :],
            W_in=W_in,
            W=W,
            W_fb=W_fb,
            bias=bias,
            feedback=feedback,
            activation=activation,
            leak_rate=leak_rate,
            noise=noise,
        )
    return states


if __name__ == "__main__":
    from functools import partial

    n_reservoir = 1000
    n_inputs = 2
    n_outputs = 2
    state = np.random.rand(n_reservoir)

    X = np.random.rand(10_000, n_inputs)
    y = np.random.rand(10_000, n_outputs)

    @njit
    def tanh(x):
        return np.tanh(x)


    W_in = np.random.rand(n_reservoir, n_inputs)
    W = np.random.rand(n_reservoir, n_reservoir)
    W_fb = np.random.rand(n_reservoir, n_outputs)
    bias = 1.2
    feedback = True
    activation = tanh
    leak_rate = .9
    noise = .001

    func = partial(
        harvest_states,
        X,
        y,
        W_in=W_in,
        W=W,
        W_fb=W_fb,
        bias=bias,
        feedback=feedback,
        activation=activation,
        leak_rate=leak_rate,
        noise=noise,
    )
    states = func()
    print(states.shape)
