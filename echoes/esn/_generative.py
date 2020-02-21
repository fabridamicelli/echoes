"""
Generative Echo State Network.
"""
import numpy as np
from sklearn.metrics import r2_score

from ._base import ESNBase
from echoes.utils import check_arrays_dimensions, check_model_params


# TODO: test initialization of inputs (should be zero)
class ESNGenerative(ESNBase):

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

        # Go through samples (steps) and predict for each of them
        for step in range(1, outputs.shape[0]):
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
            self.states_pred_ = states[1:, :]  # discard first step (comes from fitting)

        # Map outputs back to actual target space with activation function
        outputs = self.activation_out(outputs)
        return outputs[1:, :] # discard initial step (comes from fitting)

    def score(self, inputs=None, outputs=None, sample_weight=None):
        """
        From sklearn:
          R^2 (coefficient of determination) regression score function.
          Best possible score is 1.0 and it can be negative (because the model can be
          arbitrarily worse).
          A constant model that always predicts the expected value of y,
          disregarding the input features, would get a R^2 score of 0.0.

        Parameters
        ----------
        inputs: None
            Not used, present for API consistency.
            Generative ESN predicts purely based on its generative outputs.
        outputs: 2D np.ndarray of shape (n_samples, n_outputs)
            Target sequence, true values of the outputs.
        sample_weight: array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score: float
            R2 score
        """
        n_samples = outputs.shape[0]
        y_pred = self.predict(n_steps=n_samples)
        return r2_score(outputs, y_pred, sample_weight=sample_weight)
