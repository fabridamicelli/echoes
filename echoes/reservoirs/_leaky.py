"""
Reservoir of Leaky Neurons.
"""
from typing import Callable, Optional, Union

import numpy as np


class ReservoirLeakyNeurons:
    def __init__(
        self,
        W_in: np.ndarray = None,
        W: np.ndarray = None,
        W_fb: np.ndarray = None,
        bias: Union[np.ndarray, float] = 1,
        feedback: bool = False,
        activation: Callable = None,
        leak_rate: float = 1,
        noise: float = 0,
    ):
        self.W_in = W_in
        self.W = W
        self.W_fb = W_fb
        self.bias = bias
        self.activation = activation
        self.noise = noise
        self.feedback = feedback
        self.leak_rate = leak_rate
        self.n_reservoir = len(W)

    def update_state(
        self,
        state_t: np.ndarray = None,
        X_t: np.ndarray = None,
        y_t: np.ndarray = None,
    ) -> np.ndarray:
        """
        Return states vector after one time step.
        Notes: Keep this function as autonomous as possible for numba.
        """
        new_state = self.W_in @ X_t + self.W @ state_t + self.bias

        if self.feedback:
            new_state += self.W_fb @ y_t

        new_state = self.activation(new_state)

        #TODO: check noise: is -0.5 shift necessary?
        if self.noise:
            new_state += self.noise * (np.random.rand(self.n_reservoir) - 0.5)

        # Apply leakage
        if self.leak_rate < 1:
            new_state = self.leak_rate * new_state + (1 - self.leak_rate) * state_t

        return new_state

    def harvest_states(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_state: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """
        Given inputs/outputs X/y, run activity and harvest reservoir states.
        """
        n_time_steps = X.shape[0]
        states = np.zeros((n_time_steps, self.n_reservoir))
        if initial_state is not None:
            states[0, :] = initial_state

        for t in range(1, n_time_steps):
            states[t, :] = self.update_state(
                state_t=states[t - 1],
                X_t=X[t, :],
                y_t=y[t - 1, :],
            )
        return states

# def harvest_states(
    # self,
    # X: np.ndarray,
    # y: np.ndarray,
    # initial_state: Union[np.ndarray, None] = None
# ) -> np.ndarray:

    # states = leaky.harvest_states(
        # X=X,
        # y=y,
        # initial_state=initial_state,
        # W_in=self.W_in,
        # W=self.W,
        # W_fb=self.W_fb,
        # bias=self.bias,
        # feedback=self.feedback,
        # activation=self.activation,
        # leak_rate=self.leak_rate,
        # noise=self.noise,
    # )
    # return states



if __name__ == "__main__":
    n_reservoir = 1000
    n_inputs = 2
    n_outputs = 2
    state = np.random.rand(n_reservoir)

    X = np.random.rand(10_000, n_inputs)
    y = np.random.rand(10_000, n_outputs)

    from numba import njit

    @njit
    def tanh(x):
        return np.tanh(x)

    reservoir = ReservoirLeakyNeurons(
        W_in = np.random.rand(n_reservoir, n_inputs),
        W = np.random.rand(n_reservoir, n_reservoir),
        W_fb = np.random.rand(n_reservoir, n_outputs),
        bias = 1.2,
        feedback = True,
        activation = tanh,
        leak_rate = .9,
        noise = .001,
    )
    states = reservoir.harvest_states(X, y)
    print(states.shape)
