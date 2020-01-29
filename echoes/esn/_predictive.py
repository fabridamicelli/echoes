"""
Predictive Echo State Network.
"""
import numpy as np

from ._base import ESNBase
from echoes.utils import check_arrays_dimensions, check_model_params


class ESNPredictive(ESNBase):

    def predict(self, inputs):
        """
        Predict outputs according to inputs.
        State/output is reinitialized to predict test outputs from
        inputs as a typical predictive model. Since the reservoir states are
        reinitialized, an initial transient, unstable phase will occur, so you
        might want to cut off those steps to test performance (as done by the
        parameter n_transient during training).

        Parameters
        ----------
        inputs: 2D np.ndarray of shape (n_samples, n_inputs)
            Testing input, i.e., X, the features.

        Returns
        -------
        outputs: 2D np.ndarray of shape (n_samples, n_outputs)
            Predicted outputs.
        """
        n_samples = inputs.shape[0]

        # Scale and shift inputs
        inputs = self.scale_shift_inputs(inputs)

        # Append the bias to inputs -> [1; u(t)]
        bias = np.ones((n_samples, 1)) * self.bias
        inputs = np.hstack((bias, inputs))

        # Initialize predictions
        states = np.zeros((n_samples, self.n_reservoir))
        outputs = np.zeros((n_samples, self.n_outputs))
        check_arrays_dimensions(inputs)  # sanity check

        # Go through samples (steps) and predict for each of them
        for step in range(1, n_samples):
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
            self.states_pred_ = states

        # Map outputs back to actual target space with activation function
        outputs = self.activation_out(outputs)
        return outputs
