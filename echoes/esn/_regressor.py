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
    """
    Number of input and output neurons are infered from passed data.

    Arguments:
        n_reservoir: int, optional, default=100
            Number of reservoir neurons. Only used if W is not passed.
            If W is passed, n_reservoir gets overwritten with len(W).
            Either n_reservoir or W must be passed.
        W: np.ndarray of shape (n_reservoir, n_reservoir), optional, default=None
            Reservoir weights matrix. If None, random weights are used (uniformly
            distributed around 0, ie., in [-0.5, 0.5).
            Be careful with the distribution of W values. Wrong W initialization
            might drastically affect test performance (even with reasonable good
            training fit).
            Spectral radius will be adjusted in all cases.
            Either n_reservoir or W must be passed.
        spectral_radius: float, default=.99
            Spectral radius of the reservoir weights matrix (W).
            Spectral radius will be adjusted in all cases (also with user specified W).
        W_in: np.ndarray of shape (n_reservoir, 1+n_inputs) (1->bias),
            optional, default None.
            Input weights matrix by which input signal is multiplied.
            If None, random weights are used.
        W_fb: np.ndarray of shape(n_reservoir, n_outputs), optional, default None.
            Feedback weights matrix by which feedback is multiplied in case of feedback.
        sparsity: float, optional, default=0
            Proportion of the reservoir matrix weights forced to be zero.
            Note that with default W (centered around 0), the actual sparsity will
            be slightly more than the specified.
            If W is passed, sparsity will be ignored.
        noise: float, optional, default=0
            Scaling factor of the (uniform) noise input added to neurons at each step.
            This is used for regularization purposes and should typically be
            very small, e.g. 0.0001 or 1e-5.
        leak_rate: float, optional, default=1
            Leaking rate applied to the neurons at each step.
            Default is 1, which is no leaking. 0 would be total leakeage.
        bias: int, float or np.ndarray, optional, default=1
            Value of the bias neuron, injected at each time to the reservoir neurons.
            If int or float, all neurons receive the same.
            If np.ndarray is must be of length n_reservoir.
        input_scaling: float or np.ndarray of length n_inputs, default=None
            Scalar to multiply each input before feeding it to the network.
            If float, all inputs get multiplied by same value.
            If array, it must match n_inputs length (X.shape[1]), specifying the scaling
            factor for each input.
        input_shift: float or np.ndarray of length n_inputs, default=None
            Scalar to add to each input before feeding it to the network.
            If float, multiplied same value is added to all inputs.
            If array, it must match n_inputs length (X.shape[1]), specifying the value
            to add to each input.
        feedback: bool, optional, default=False
            If True, the reservoir also receives the outout signal as input.
        activation: function (numba jitted), optional, default=tanh
            Non-linear activation function applied to the neurons at each step.
            For numba acceleration, it must be a jitted function.
            Basic activation functions as tanh, sigmoid, relu are already available
            in echoe.utils. Either use those or write a custom one decorated with
            numba njit.
        activation_out: function, optional, default=identity
            Activation function applied to the outputs. In other words, it is assumed
            that targets = f(outputs). So the output produced must be transformed.
        fit_only_states: bool,default=False
            If True, outgoing weights (W_out) are computed fitting only the reservoir
            states. Inputs and bias are still use to drive reservoir activity, but
            ignored for fitting W_out, both in the training and prediction phase.
        regression_method: str, optional, default "pinv" (pseudoinverse).
            Method to solve the linear regression to find out outgoing weights.
            One of ["pinv", "ridge"].
            If "ridge", ridge_* parameters will be used.
        ridge_alpha: float, ndarray of shape (n_outputs,), default=1
            Regularization coefficient used for Ridge regression.
            Larger values specify stronger regularization.
            If an array is passed, penalties are assumed to be specific to the targets.
            Hence they must correspond in number.
            Default is None to make sure one deliberately sets this since it is
            a crucial parameter. See sklearn Ridge documentation for details.
        ridge_fit_intercept: bool, optional, default=False
            If True, intercept is fit in Ridge regression. Default False.
            See sklearn Ridge documentation for details.
        ridge_max_iter: int, default=None
            Maximum number of iterations for conjugate gradient solver.
            See sklearn Ridge documentation for details.
        ridge_tol: float, default=1e-3
            Precision of the solution.
            See sklearn Ridge documentation for details.
        ridge_solver: str, optional, default="auto"
            Solver to use in the Ridge regression.
            One of ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"].
            See sklearn Ridge documentation for details.
        ridge_sample_weight: float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample.
            If given a float, every sample will have the same weight.
            See sklearn Ridge documentation for details.
        n_transient: int, optional, default=0
            Number of activity initial steps removed (not considered for training)
            in order to avoid initial instabilities.
            Default is 0, but this is something one definitely might want to tweak.
        random_state : int, RandomState instance, default=None
            The seed of the pseudo random number generator used to generate weight
            matrices, to generate noise inyected to reservoir neurons (regularization)
            and it is passed to the ridge solver in case regression_method=ridge.
            From sklearn:
              If int, random_state is the seed used by the random number generator;
              If RandomState instance, random_state is the random number generator;
              If None, the random number generator is the RandomState instance used
              by `np.random`.
        store_states_train: bool, optional, default=False
            If True, time series series of reservoir neurons during training are stored
            in the object attribute states_train_.
        store_states_pred: bool, optional, default=False
            If True, time series series of reservoir neurons during prediction are
            stored in the object attribute states_pred_.

    ### Attributes:
        - W_out_ : array of shape (n_outputs, n_inputs + n_reservoir + 1)
             Outgoing weights after fitting linear regression model to predict outputs.
        - training_prediction_: array of shape (n_samples, n_outputs)
             Predicted output on training data.
        - states_train_: array of shape (n_samples, n_reservoir), default False.
             If store_states_train is True, states matrix is stored for visualizing
             reservoir neurons activity during training.
        - states_pred_: array of shape (n_samples, n_reservoir), default False.
             If store_states_pred is True, states matrix is stored for visualizing
             reservoir neurons activity during prediction (test).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ESNRegressor":
        """
        Fit Echo State model, i.e., find outgoing weights matrix (W_out) for later
        prediction.
        Bias is appended automatically to the inputs.

        Arguments:
            X: None or 2D np.ndarray of shape (n_samples, n_inputs)
                Training input, i.e., X, the features.
                If None, it is assumed that only the target sequence matters (outputs)
                and simply a sequence of zeros will be fed in - matching the
                len(outputs).
                This is to be used in the case of generative mode.
            y: 2D np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
                Training output, i.e., y, the target.

        Returns
            self: returns an instance of self.
        """
        self._dtype_ = X.dtype
        X, y = check_X_y(X, y, multi_output=True, dtype=self._dtype_)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # Check y again (enforcing 2D for multiple outputs)
        y = check_array(y, dtype=self._dtype_)

        # Initialize matrices and random state
        self.random_state_ = check_random_state(self.random_state)
        self.n_inputs_ = X.shape[1]
        self.n_reservoir_ = len(self.W) if self.W is not None else self.n_reservoir
        self.n_outputs_ = y.shape[1]
        self.W_in_ = self._init_incoming_weights()
        self.W_ = self._init_reservoir_weights()
        self.W_fb_ = self._init_feedback_weights()

        check_model_params(self.__dict__)
        X = self._scale_shift_inputs(X)

        # Initialize reservoir model
        self.reservoir_ = self._init_reservoir_neurons()

        # Run "neuronal activity"
        states = self.reservoir_.harvest_states(X, y, initial_state=None)

        # Extend states matrix with inputs, except we only train based on states
        full_states = states if self.fit_only_states else np.hstack((states, X))

        # Solve for W_out using full states and outputs, excluding transient
        self.W_out_ = self._solve_W_out(
            full_states[self.n_transient :, :], y[self.n_transient :, :]
        )
        # Predict on training set (including the pass through the output nonlinearity)
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
                Input, i.e., X, the features.

        Returns:
            y_pred: 2D np.ndarray of shape (n_samples, n_outputs)
                Predicted outputs.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=self._dtype_)

        n_time_steps = X.shape[0]

        # Scale and shift inputs
        X = self._scale_shift_inputs(X)

        # Initialize predictions
        states = np.zeros((n_time_steps, self.n_reservoir_), dtype=self._dtype_)
        y_pred = np.zeros((n_time_steps, self.n_outputs_), dtype=self._dtype_)

        check_consistent_length(X, y_pred)  # sanity check

        # Go through samples (steps) and predict for each of them
        for t in range(1, n_time_steps):
            states[t, :] = self.reservoir_.update_state(
                state_t=states[t - 1, :],
                X_t=X[t, :],
                y_t=y_pred[t - 1, :],
            )

            if self.fit_only_states:
                full_states = states[t, :]
            else:
                full_states = np.concatenate([states[t, :], X[t, :]])
            # Predict
            y_pred[t, :] = self.W_out_ @ full_states

        # Store reservoir activity
        if self.store_states_pred:
            self.states_pred_ = states

        # Apply output non-linearity
        return self.activation_out(y_pred)

    def score(self, X: np.ndarray, y=np.ndarray, sample_weight=None) -> float:
        """
        R^2 (coefficient of determination) regression score function.

        By default, the initial transient period (n_transient steps) is not considered
        to compute the score - modify sample_weight to change that behaviour
        (see below).

        From sklearn:
          Best possible score is 1.0 and it can be negative (because the model can be
          arbitrarily bad).
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
                To consider all steps or leave out a different transient, pass a
                different sample_weight array with same length as outputs 1 dimension.
                **Usage**
                  >> n_steps_to_remove = 10
                  >> weights = np.ones(y_true.shape[0])
                  >> weights[: n_steps_to_remove] = 0
                  >> score(X, y_true, sample_weight=weights)

        Returns:
            score: float
                R2 score
        """
        y_pred = self.predict(X)
        # If no sample_weight passed, compute the score without considering transient
        if sample_weight is None:
            weights = np.ones(y.shape[0])
            weights[: self.n_transient] = 0
            return r2_score(y, y_pred, sample_weight=weights)

        return r2_score(y, y_pred, sample_weight=sample_weight)
