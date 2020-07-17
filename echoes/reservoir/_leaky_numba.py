"""
Reservoir of Leaky Neurons.

Implementation accelerated with numba.
"""
from typing import Callable, Optional, Union

from numba import njit, jit
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
        feedback: bool, optional, default=False
            If True, the reservoir also receives the outout signal as input.
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
        W_in: np.ndarray = None,
        W: np.ndarray = None,
        W_fb: np.ndarray = None,
        feedback: bool = False,
        bias: Union[np.ndarray, float] = 1,
        activation: Callable = None,
        noise: float = 0,
        leak_rate: float = 1,
    ):
        self.W_in = W_in
        self.W = W
        self.W_fb = W_fb
        self.feedback = feedback
        self.bias = bias
        self.activation = activation
        self.noise = noise
        self.leak_rate = leak_rate
        self.n_reservoir = len(W)

    def update_state(
        self,
        state_t: np.ndarray = None,
        X_t: np.ndarray = None,
        y_t: np.ndarray = None,
    ) -> np.ndarray:
        new_state = update_state(
            state_t=state_t,
            X_t=X_t,
            y_t=y_t,
            W_in=self.W_in,
            W=self.W,
            W_fb=self.W_fb,
            feedback=self.feedback,
            bias=self.bias,
            activation=self.activation,
            noise=self.noise,
            leak_rate=self.leak_rate
        )
        return new_state

    def harvest_states(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_state: Union[np.ndarray, None] = None
    ) -> np.ndarray:

        states = harvest_states(
            X,
            y,
            initial_state=initial_state,
            W_in=self.W_in,
            W=self.W,
            W_fb=self.W_fb,
            feedback=self.feedback,
            bias=self.bias,
            activation=self.activation,
            noise=self.noise,
            leak_rate=self.leak_rate
        )
        return states

#@njit
def update_state(
    state_t: np.ndarray = None,
    X_t: np.ndarray = None,
    y_t: np.ndarray = None,
    W_in: np.ndarray = None,
    W: np.ndarray = None,
    W_fb: np.ndarray = None,
    feedback: bool = False,
    bias: np.ndarray = None,
    activation: Callable = None,
    noise: float = 0,
    leak_rate: float = 1,
) -> np.ndarray:
    """
    Return states vector after one time step.
    """
    n_reservoir = len(W)

    new_state = W_in @ X_t + W @ state_t + bias


    if not feedback:  # hack for numba, otherwise type(W_fb) is None and cannot compile
        W_fb = np.zeros_like(y_t)
    else:
        new_state += W_fb @ y_t

    new_state = activation(new_state)

    #TODO: check noise: is -0.5 shift necessary?
    if noise:
        new_state += noise * (np.random.rand(n_reservoir) - 0.5)

    # Apply leakage
    if leak_rate < 1:
        new_state = leak_rate * new_state + (1 - leak_rate) * state_t

    return new_state

#@njit
def harvest_states(
    X: np.ndarray,
    y: np.ndarray,
    initial_state: Union[np.ndarray, None] = None,
    W_in: np.ndarray = None,
    W: np.ndarray = None,
    W_fb: np.ndarray = None,
    feedback: bool = False,
    bias: np.ndarray = None,
    activation: Callable = None,
    noise: float = 0,
    leak_rate: float = 1,
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
            feedback=feedback,
            bias=bias,
            activation=activation,
            noise=noise,
            leak_rate=leak_rate
        )
    return states


if __name__ == "__main__":
    n_reservoir = 1000
    n_inputs = 2
    n_outputs = 2
    state = np.random.rand(n_reservoir)

    X = np.random.rand(20_000, n_inputs)
    y = np.random.rand(20_000, n_outputs)

    from numba import njit

    @njit
    def tanh(x):
        return np.tanh(x)

    reservoir = ReservoirLeakyNeurons(
        W_in = np.random.rand(n_reservoir, n_inputs),
        W = np.random.rand(n_reservoir, n_reservoir),
        # W_fb = np.random.rand(n_reservoir, n_outputs),
        bias = 1.2,
        feedback = False,
        activation = tanh,
        leak_rate = .9,
        noise = .001,
    )
    states = reservoir.harvest_states(X, y)
    print(states.shape)
