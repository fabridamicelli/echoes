"""
Echo State Network Generator (pattern generator).
"""

from typing import Callable, Union
import warnings

import numpy as np
from sklearn.utils.validation import (
    check_random_state,
    check_consistent_length,
)
from sklearn.utils import check_array
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.metrics import r2_score

from ._base import ESNBase
from echoes.utils import check_model_params, tanh, identity


class ESNGenerator(ESNBase, MultiOutputMixin, RegressorMixin):
    """
    The number of inputs (n_inputs) is always 1 and n_outputs is infered from passed
    data.
    It uses always feedback, so that is not a parameter anymore (always True).

    Arguments:
        n_steps: int, default=100
            Number of steps to generate pattern (used by predict method).
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
        W_in: np.ndarray of shape (n_reservoir, 1+n_inputs) (1->bias), optional,
            default None.
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
        ridge_alpha: float, ndarray of shape (n_outputs,), default=None
            Regularization coefficient used for Ridge regression.
            Larger values specify stronger regularization.
            If an array is passed, penalties are assumed to be specific to the targets.
            Hence they must correspond in number.
            Default is None to make sure one deliberately sets this since it is
            a crucial parameter. See sklearn Ridge documentation for details.
        ridge_fit_intercept: bool, optional, default=False
            If True, intercept is fit in Ridge regression. Default False.
            See sklearn Ridge documentation for details.
        ridge_normalize: bool, default=False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by
            subtracting the mean and dividing by the l2-norm.
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
        - W_out_ : array of shape (n_outputs, n_inputs + n_reservoir + 1).
            Outgoing weights after fitting linear regression model to predict outputs.
        - training_prediction_: array of shape (n_samples, n_outputs).
            Predicted output on training data.
        - states_train_: array of shape (n_samples, n_reservoir), default False.
            If store_states_train is True, states matrix is stored for visualizing
            reservoir neurons activity during training.
        - states_pred_: array of shape (n_samples, n_reservoir), default False.
            If store_states_pred is True, states matrix is stored for visualizing
            reservoir neurons activity during prediction (test).
    """

    def __init__(
        self,
        *,
        n_steps: int = 100,
        n_reservoir: int = 100,
        W: np.ndarray = None,
        spectral_radius: float = 0.99,
        W_in: np.ndarray = None,
        W_fb: np.ndarray = None,
        sparsity: float = 0,
        noise: float = 0,
        leak_rate: float = 1,
        bias: Union[int, float, np.ndarray] = 1,
        input_scaling: Union[float, np.ndarray] = None,
        input_shift: Union[float, np.ndarray] = None,
        activation: Callable = tanh,
        activation_out: Callable = identity,
        fit_only_states: bool = False,
        regression_method: str = "pinv",
        ridge_alpha: float = 1,
        ridge_fit_intercept: bool = False,
        ridge_normalize: bool = False,
        ridge_max_iter: int = None,
        ridge_tol: float = 1e-3,
        ridge_solver: str = "auto",
        ridge_sample_weight: Union[float, np.ndarray] = None,
        n_transient: int = 0,
        store_states_train: bool = False,
        store_states_pred: bool = False,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> None:
        self.n_steps = n_steps
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.W = W
        self.W_in = W_in
        self.W_fb = W_fb
        self.sparsity = sparsity
        self.noise = noise
        self.leak_rate = leak_rate
        self.bias = bias
        self.input_scaling = input_scaling
        self.input_shift = input_shift
        self.activation = activation
        self.activation_out = activation_out
        self.fit_only_states = fit_only_states
        self.n_transient = n_transient
        self.store_states_train = store_states_train
        self.store_states_pred = store_states_pred
        self.regression_method = regression_method
        self.ridge_alpha = ridge_alpha
        self.ridge_fit_intercept = ridge_fit_intercept
        self.ridge_normalize = ridge_normalize
        self.ridge_max_iter = ridge_max_iter
        self.ridge_tol = ridge_tol
        self.ridge_solver = ridge_solver
        self.ridge_sample_weight = ridge_sample_weight
        self.random_state = random_state
        self.feedback = True  # Generator uses feedback always

    def fit(self, X=None, y=None) -> "ESNGenerator":
        """
        Fit Echo State model, i.e., find outgoing weights matrix (W_out) for later
        prediction.
        Bias is appended automatically to the inputs.

        Arguments:
            X: None, always ignored, API consistency
                It is ignored as only the target sequence matters (outputs).
                A sequence of zeros will be fed in - matching the len(outputs) as
                initial condition.
            y: 2D np.ndarray of shape (n_samples,) or (n_samples, n_outputs),
                default=None.
                Target variable.

        Returns:
            self: returns an instance of self.
        """
        if X is not None:
            warnings.warn("X will be ignored – ESNGenerator only takes y for training")
        y = check_array(y, ensure_2d=False, dtype=np.float64)
        self._dtype_ = y.dtype
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        outputs = y

        # Initialize matrices and random state
        self.random_state_ = check_random_state(self.random_state)
        # Pattern generation takes no input, thus hardcode for later
        # construction of matrices
        self.n_inputs_ = 1
        self.n_reservoir_ = len(self.W) if self.W is not None else self.n_reservoir
        self.n_outputs_ = outputs.shape[1]
        self.W_in_ = self._init_incoming_weights()
        self.W_ = self._init_reservoir_weights()
        self.W_fb_ = self._init_feedback_weights()

        check_model_params(self.__dict__)

        # Make inputs zero
        inputs = np.zeros(shape=(outputs.shape[0], self.n_inputs_), dtype=self._dtype_)

        check_consistent_length(inputs, outputs)  # sanity check

        # Initialize reservoir model
        self.reservoir_ = self._init_reservoir_neurons()

        states = self.reservoir_.harvest_states(inputs, outputs, initial_state=None)

        # Extend states matrix with inputs; i.e., make [h(t); x(t)]
        full_states = states if self.fit_only_states else np.hstack((states, inputs))

        # Solve for W_out using full states and outputs, excluding transient
        self.W_out_ = self._solve_W_out(
            full_states[self.n_transient :, :], outputs[self.n_transient :, :]
        )
        # Predict on training set (including the pass through the output nonlinearity)
        self.training_prediction_ = self.activation_out(full_states @ self.W_out_.T)

        # Keep last state for later
        self.last_state_ = states[-1, :]
        self.last_input_ = inputs[-1, :]
        self.last_output_ = outputs[-1, :]

        # Store reservoir activity
        if self.store_states_train:
            self.states_train_ = states
        return self

    def predict(self, X=None) -> np.ndarray:
        """
        Last training state/input/output is used as initial test
        state/input/output and at each step the output of the network is reinjected
        as input for next prediction, thus no inputs are needed for prediction.

        Arguments:
            X: None, always ignored, API consistency

        Returns:
            outputs: 2D np.ndarray of shape (n_steps, n_outputs)
             Predicted outputs.
        """
        if X is not None:
            warnings.warn("X will be ignored – ESNGenerator takes no X for prediction")

        assert self.n_steps >= 1, "n_steps must be >= 1"
        n_steps = self.n_steps  # shorthand

        # Initialize predictions: begin with last state as first state
        inputs = np.zeros(shape=(n_steps, self.n_inputs_), dtype=self._dtype_)
        inputs = np.vstack([self.last_input_, inputs])
        states = np.vstack([
            self.last_state_,
            np.zeros((n_steps, self.n_reservoir_), dtype=self._dtype_),
        ])
        outputs = np.vstack([
            self.last_output_,
            np.zeros((n_steps, self.n_outputs_), dtype=self._dtype_),
        ])

        check_consistent_length(inputs, outputs)  # sanity check

        # Go through samples (steps) and predict for each of them
        for t in range(1, outputs.shape[0]):
            states[t, :] = self.reservoir_.update_state(
                state_t=states[t - 1, :],
                X_t=inputs[t, :],
                y_t=outputs[t - 1, :],
            )
            if self.fit_only_states:
                full_states = states[t, :]
            else:
                full_states = np.concatenate([states[t, :], inputs[t, :]])
            # Predict
            outputs[t, :] = self.W_out_ @ full_states

            # TODO: Update last_{input, states, outputs}_

        # Store reservoir activity
        if self.store_states_pred:
            self.states_pred_ = states[1:, :]  # discard first step (comes from fitting)

        # Apply output non-linearity
        outputs = self.activation_out(outputs)
        return outputs[1:, :]  # discard initial step (comes from training)

    def score(self, X=None, y=None, sample_weight=None) -> float:
        """
        Wrapper around sklearn r2_score with kwargs.

        From sklearn:
          R^2 (coefficient of determination) regression score function.
          Best possible score is 1.0 and it can be negative (because the model can be
          arbitrarily worse).
          A constant model that always predicts the expected value of y,
          disregarding the input features, would get a R^2 score of 0.0.

        Arguments:
            X: None
                Not used, present for API consistency.
                Generative ESN predicts purely based on its generative outputs.
            y: 2D np.ndarray of shape (n_samples, ) or (n_samples, n_outputs)
                Target sequence, true values of the outputs.
            sample_weight: array-like of shape (n_samples,), default=None
                Sample weights.

        Returns:
            score: float
                R2 score
        """
        return r2_score(y, self.predict(), sample_weight=sample_weight)
