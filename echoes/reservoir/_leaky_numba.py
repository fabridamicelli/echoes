"""
Reservoir of Leaky Neurons.

Implementation accelerated with numba.
"""

from __future__ import annotations  # TODO: Remove after dropping python 3.9
from typing import Callable

from numba import njit
import numpy as np


class ReservoirLeakyNeurons:
    """
    Reservoir of leaky neurons (see Notes for equations).

    Arguments:
        W_in: np.ndarray of shape (n_reservoir, 1+n_inputs) (1->bias), optional, default None.
            Input weights matrix by which input signal is multiplied.
        W: np.ndarray of shape (n_reservoir, n_reservoir)
            Reservoir weights matrix.
        W_fb: np.ndarray of shape(n_reservoir, n_outputs), optional, default None.
            Feedback weights matrix by which feedback is multiplied in case of feedback.
        bias: int, float or np.ndarray, optional, default=1
            Value of the bias neuron, injected at each time to the reservoir neurons.
            If int or float, all neurons receive the same.
            If np.ndarray is must be of length n_reservoir.
        activation: function (numba jitted)
            Non-linear activation function applied to the neurons at each step.
            For numba acceleration, it must be a jitted function.
            Basic activation functions as tanh, sigmoid, relu are already available
            in echoe.utils. Either use those or write a custom one decorated with
            numba njit.
        leak_rate: float, optional, default=1
            Leaking rate applied to the neurons at each step.
            Default is 1, which is no leaking. 0 would be total leakeage.
        noise: float, optional, default=0
            Scaling factor of the (uniform) noise input added to neurons at each step.
            This is used for regularization purposes and should typically be
            very small, e.g. 0.0001 or 1e-5.

    Notes:
        Reservoir of leaky neurons, where the actvity of the neurons is governed by the
        following equations:

        1) h'(t) = f(W_in @ X(t) + W @ h(t-1) + W_fb @ y(t-1) + b) + noise
        2) h(t)  = (1-a) * h(t-1) + a * h'(t)

        Where:
          @: dot product
          h'(t): reservoir states at time t before applying leakeage
          h(t): reservoir (hidden) states vector at time t after applying leakeage
          a: leak rate (1 is no leakeage, 0 is complete leakeage)
          f: activation function
          noise: random noise applied to neurons (regularization)
          W: reservoir weights matrix
          W_in: incoming weights matrix
          W_fb: feedback matrix
          X(t): inputs vector at time t
          y(t): outputs vector at time t
          b: bias vector applied to the reservoir neurons
    """

    def __init__(
        self,
        *,
        W_in: np.ndarray,
        W: np.ndarray,
        W_fb: np.ndarray,
        bias: np.ndarray | float = 1.0,
        activation: Callable,
        noise: float = 0.0,
        leak_rate: float = 1.0,
    ):
        _dtype = W.dtype
        self.W_in = W_in.astype(_dtype)
        self.W = W
        self.W_fb = W_fb.astype(_dtype) if W_fb is not None else W_fb
        self.bias = np.array(bias).astype(_dtype)
        self.activation = activation
        self.noise = np.array(noise).astype(_dtype)
        self.leak_rate = np.array(leak_rate).astype(_dtype)
        self.n_reservoir = len(W)

        types = set(
            param.dtype
            for param in (
                self.W_in,
                self.W,
                self.W_fb,
                self.bias,
                self.noise,
                self.leak_rate,
            )
            if param is not None
        )

        assert len(types) == 1, "type inconsistency"

    def update_state(
        self,
        *,
        state_t: np.ndarray,
        X_t: np.ndarray,
        y_t: np.ndarray,
    ) -> np.ndarray:
        new_state = update_state(
            state_t=state_t,
            X_t=X_t,
            y_t=y_t,
            W_in=self.W_in,
            W=self.W,
            W_fb=self.W_fb,
            bias=self.bias,
            activation=self.activation,
            noise=self.noise,
            leak_rate=self.leak_rate,
        )
        return new_state

    def harvest_states(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        states = harvest_states(
            X=X,
            y=y,
            initial_state=initial_state,
            W_in=self.W_in,
            W=self.W,
            W_fb=self.W_fb,
            bias=self.bias,
            activation=self.activation,
            noise=self.noise,
            leak_rate=self.leak_rate,
        )
        return states


@njit
def update_state(
    state_t: np.ndarray,
    X_t: np.ndarray,
    y_t: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    W_fb: np.ndarray,
    bias: np.ndarray,
    activation: Callable,
    noise: float = 0.0,
    leak_rate: float = 1.0,
) -> np.ndarray:
    """
    Return states vector after one time step.
    """
    new_state = activation(W_in @ X_t + W @ state_t + W_fb @ y_t + bias)

    # TODO: check noise: is -0.5 shift necessary?
    if noise > 0:
        new_state += noise * (np.random.rand(W.shape[0]) - 0.5)

    # Apply leakage
    if leak_rate < 1:
        new_state = leak_rate * new_state + (1 - leak_rate) * state_t

    return new_state


@njit
def harvest_states(
    *,
    X: np.ndarray,
    y: np.ndarray,
    initial_state: np.ndarray | None = None,
    W_in: np.ndarray,
    W: np.ndarray,
    W_fb: np.ndarray,
    bias: np.ndarray,
    activation: Callable,
    noise: float = 0.0,
    leak_rate: float = 1.0,
) -> np.ndarray:
    """
    Given inputs/outputs X/y, run activity and harvest reservoir states.
    """
    n_reservoir = len(W)
    n_time_steps = X.shape[0]
    states = np.zeros((n_time_steps, n_reservoir), dtype=X.dtype)
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
            activation=activation,
            noise=noise,
            leak_rate=leak_rate,
        )
    return states
