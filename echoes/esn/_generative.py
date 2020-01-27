"""
Generative Echo State Network class.
"""
import warnings

import numpy as np

from ._base import ESNBase
from ..utils import check_arrays_dimensions


class ESNGenerative(ESNBase):
    def __init__(self,):
        # TODO:
        # warnings: input shift and scaling will be ignored in generative mode
        # only bias will have a contribution
        assert self.teacher_forcing == True, "generative mode requires teacher forcing"

    def predict(self, n_steps=None):
        """
        Predict according to inputs and mode.
        Last training state/input/output is used as initial test
        state/input/output and at each step the output of the network is reinjected
        as input for next prediction.

        Parameters
        ----------
        n_steps: int
            Number of generative steps to predict.

        Returns
        -------
        outputs: 2D np.ndarray of shape (n_steps, n_outputs)
            Predicted outputs.
        """
        assert n_steps >= 1, "n_steps must be >= 1"

        # Scale and shift inputs are ignored, since only bias should contribute to
        # next state and inputs should be zero
        inputs = np.zeros(shape=(n_steps, self.n_inputs))
        check_arrays_dimensions(inputs)

        # Append the bias to inputs -> [1; u(t)]
        bias = np.ones((n_steps, 1)) * self.bias
        inputs = np.hstack((bias, inputs))

        # Initialize predictions: begin with last state as first state
        inputs = np.vstack([self.last_input, inputs])
        states = np.vstack([self.last_state, np.zeros((n_steps, self.n_reservoir))])
        outputs = np.vstack([self.last_output, np.zeros((n_steps, self.n_outputs))])

        assert np.all(inputs == 0), "wrong inputs initialization"

        # Go through samples (steps) and predict for each of them
        for step in range(1, n_steps):
            states[step, :] = self._update_state(
                states[step - 1, :], inputs[step, :], outputs[step - 1, :]
            )

            if self.fit_only_states:
                full_states = states[step, :]
            else:
                full_states = np.concatenate([states[step, :], inputs[step, :]])
            # Predict
            outputs[step, :] = self.W_out_ @ full_states

        # Store reservoir activity
        if self.store_states_pred:
            self.states_pred_ = states[1:, :]  # discard initial step (comes from fitting)

        # Map outputs back to actual target space with activation function
        outputs = self.activation_out(outputs)
        return outputs[1:, :] # discard initial step (comes from fitting)
