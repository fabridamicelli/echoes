"""
Echo State Network Regressor.
"""
import numpy as np
from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    check_consistent_length,
)
from sklearn.utils import check_X_y, check_array
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.metrics import r2_score

from ._base import ESNBase
from echoes.utils import check_model_params


class ESNRegressor(ESNBase, MultiOutputMixin, RegressorMixin):

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ESNRegressor":
        """
        Fit Echo State model, i.e., find outgoing weights matrix (W_out) for later
        prediction.
        Bias is appended automatically to the inputs.

        Arguments:
            X: None or 2D np.ndarray of shape (n_samples, n_inputs)
                Training input, i.e., X, the features.
                If None, it is assumed that only the target sequence matters (outputs)
                and simply a sequence of zeros will be fed in - matching the len(outputs).
                This is to be used in the case of generative mode.
            y: 2D np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
                Training output, i.e., y, the target.

        Returns
            self: returns an instance of self.
        """
        X, y = check_X_y(X, y, multi_output=True)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        inputs, outputs = X, y

        # Initialize matrices and random state
        self.random_state_ = check_random_state(self.random_state)
        self.n_inputs_ = inputs.shape[1]
        self.n_reservoir_ = len(self.W) if self.W is not None else self.n_reservoir
        self.n_outputs_ = outputs.shape[1]
        self.W_in_ = self._init_incoming_weights()
        self.W_ = self._init_reservoir_weights()
        self.W_fb_ = self._init_feedback_weights()

        check_model_params(self.__dict__)
        #######--#####
        # Scale and shift inputs
        inputs = self._scale_shift_inputs(inputs)

        # Inverse transform outputs (map them into inner, latent space)
        outputs = self.inv_activation_out(outputs)

        n_samples = inputs.shape[0]
        # Append the bias to inputs -> [1; u(t)]
        bias = np.ones((n_samples, 1)) * self.bias
        inputs = np.hstack((bias, inputs))
        # Collect reservoir states through the given input,output pairs
        states = np.zeros((n_samples, self.n_reservoir_))
        for step in range(1, n_samples):
            states[step, :] = self._update_state(
                states[step - 1],
                inputs[step, :],
                outputs[step - 1, :],
                self.W_in_,
                self.W_,
                self.W_fb_,
            )

        # Extend states matrix with inputs (and bias); i.e., make [x(t); 1; u(t)]
        full_states = states if self.fit_only_states else np.hstack((states, inputs))

        # Solve for W_out using full states and outputs, excluding transient
        self.W_out_ = self._solve_W_out(
            full_states[self.n_transient :, :], outputs[self.n_transient :, :]
        )
        # Predict on training set (map them back to original space with activation)
        self.training_prediction_ = self.activation_out(full_states @ self.W_out_.T)

        # Store reservoir activity
        if self.store_states_train:
            self.states_train_ = states
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs according to inputs.
        State/output is reinitialized to predict test outputs from
        inputs as a typical predictive model. Since the reservoir states are
        reinitialized, an initial transient, unstable phase will occur, so you
        might want to cut off those steps to test performance (as done by the
        parameter n_transient during training).

        Arguments:
            X: 2D np.ndarray of shape (n_samples, n_inputs)
                Testing input, i.e., X, the features.

        Returns:
            outputs: 2D np.ndarray of shape (n_samples, n_outputs)
                Predicted outputs.
        """
        check_is_fitted(self)
        X = check_array(X)

        inputs = X
        n_samples = inputs.shape[0]

        # Scale and shift inputs
        inputs = self._scale_shift_inputs(inputs)

        # Append the bias to inputs -> [1; u(t)]
        bias = np.ones((n_samples, 1)) * self.bias
        inputs = np.hstack((bias, inputs))

        # Initialize predictions
        states = np.zeros((n_samples, self.n_reservoir_))
        outputs = np.zeros((n_samples, self.n_outputs_))

        check_consistent_length(inputs, outputs)  # sanity check

        # Go through samples (steps) and predict for each of them
        for step in range(1, n_samples):
            states[step, :] = self._update_state(
                states[step - 1, :],
                inputs[step, :],
                outputs[step - 1, :],
                self.W_in_,
                self.W_,
                self.W_fb_,
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

    # TODO: handle transient
    def score(self, X=None, y=None, sample_weight=None) -> float:
        """
        R^2 (coefficient of determination) regression score function.

        By default, the initial transient period (n_transient steps) is not considered
        to compute the score - modify sample_weight to change that behaviour (see below).

        From sklearn:
          Best possible score is 1.0 and it can be negative (because the model can be
          arbitrarily worse).
          A constant model that always predicts the expected value of y,
          disregarding the input features, would get a R^2 score of 0.0.

        Arguments:
            X: 2D np.ndarray of shape (n_samples, n_inputs)
                Test samples.
            y: 2D np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
                Target sequence, true values of the outputs.
            sample_weight: array-like of shape (n_samples,), default=None
                Sample weights.
                If None, the transient is left out.
                To consider all steps or leave out a different transient, pass a different
                sample_weight array with same length as outputs 1 dimension.
                Example:
                  >> n_steps_to_remove = 10
                  >> weights = np.ones(outputs.shape[0])
                  >> weights[: n_steps_to_remove] = 0
                  >> score(inputs, outputs, sample_weight=weights)

        Returns:
            score: float
                R2 score
        """
        y_pred = self.predict(X)
        if sample_weight is None:
            weights = np.ones(y.shape[0])
            weights[: self.n_transient] = 0
            return r2_score(y, y_pred, sample_weight=weights)

        return r2_score(y, y_pred, sample_weight=sample_weight)
