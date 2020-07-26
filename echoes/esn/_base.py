"""
Echo State Network (ESN) estimator base class.
It implements common code for ESN estimators (ESNRegressor, ESNGenerator).
It should not be instanciated.
"""
from typing import Union, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from echoes.utils import set_spectral_radius, identity, tanh
from echoes.reservoir import ReservoirLeakyNeurons


class ESNBase(BaseEstimator):
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
        W_in: np.ndarray of shape (n_reservoir, 1+n_inputs) (1->bias), optional, default None.
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
            If array, it must match n_inputs length (X.shape[1]), specifying the value to
            add to each input.
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
            If True, time series series of reservoir neurons during prediction are stored
            in the object attribute states_pred_.

    Attributes:
        W_out_ : array of shape (n_outputs, n_inputs + n_reservoir + 1)
            Outgoing weights after fitting linear regression model to predict outputs.
        training_prediction_: array of shape (n_samples, n_outputs)
            Predicted output on training data.
        states_train_: array of shape (n_samples, n_reservoir), default False.
            If store_states_train is True, states matrix is stored for visualizing
            reservoir neurons activity during training.
        states_pred_: array of shape (n_samples, n_reservoir), default False.
            If store_states_pred is True, states matrix is stored for visualizing
            reservoir neurons activity during prediction (test).
    """

    def __init__(
        self,
        *,
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
        feedback: bool = False,
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
        self.feedback = feedback
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

    def _init_incoming_weights(self) -> np.ndarray:
        """
        Return matrix W_in of random incoming weights.
        Shape (n_reservoir, n_inputs+1), where +1 corresponds to bias column.
        Note: bias and input weights are not initialized separately.
        # TODO: initialize bias weights separately, as we might want bias to have
                a different contribution than the inputs.
        """
        if self.W_in is not None:
            return self.W_in
        W_in = self.random_state_.uniform(
            low=-1, high=1, size=(self.n_reservoir_, self.n_inputs_)
        )
        return W_in.astype(self._dtype_)

    def _init_reservoir_weights(self) -> np.ndarray:
        """
        Return reservoir weights matrix W. Shape (n_reservoir, n_reservoir).
        Sparsity and spectral_radius are adjusted.
        Init random matrix centered around zero with desired spectral radius
        """
        if self.W is not None:
            return set_spectral_radius(self.W, self.spectral_radius)

        W = self.random_state_.uniform(
            low=-0.5, high=0.5, size=(self.n_reservoir_, self.n_reservoir_)
        )
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        return set_spectral_radius(W, self.spectral_radius).astype(self._dtype_)

    def _init_reservoir_neurons(self) -> "ReservoirLeakyNeurons":
        """
        Instantiate the dynamical model that governs reservoir activity.
        The returned reservoir will be used to update and harvest reservoir states.
        """
        reservoir = ReservoirLeakyNeurons(
            W_in=self.W_in_,
            W=self.W_,
            W_fb=self.W_fb_,
            bias=self.bias,
            activation=self.activation,
            leak_rate=self.leak_rate,
            noise=self.noise,
        )
        return reservoir

    def _init_feedback_weights(self) -> Union[None, np.ndarray]:
        """Return feedback weights. Shape (n_reservoir, n_outputs)."""
        if not self.feedback:
            return np.zeros((self.n_reservoir_, self.n_outputs_), dtype=self._dtype_)
        if self.W_fb is not None:
            return self.W_fb
        W_fb = self.random_state_.uniform(
            low=-1, high=1, size=(self.n_reservoir_, self.n_outputs_)
        )
        return W_fb.astype(self._dtype_)

    # TODO maybe move to utils and/or replace by sklearn scaler
    def _scale_shift_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Return first scaled and then shifted inputs vector/matrix.
        """
        if self.input_scaling is not None:
            if isinstance(self.input_scaling, (float, int)):
                inputs = inputs * self.input_scaling
            elif isinstance(self.input_scaling, np.ndarray):
                assert len(self.input_scaling) == self.n_inputs_, "wrong input scaling"
                inputs = inputs * self.input_scaling
            else:
                raise ValueError("wrong input scaling type")

        if self.input_shift is not None:
            if isinstance(self.input_shift, (float, int)):
                inputs = inputs + self.input_shift
            elif isinstance(self.input_scaling, np.ndarray):
                assert len(self.input_shift) == self.n_inputs_, "wrong input shift"
                inputs = inputs + self.input_scaling
            else:
                raise ValueError("wrong input scaling type")

        return inputs

    def _solve_W_out(self, full_states: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """
        Solve for outgoing weights with linear regression, i.e., the equation:
                W_out = Y X.T inv(X X.T)

        Solution is found according to regression_method.

        Parameters
        ----------
        full_states: 2D np.ndarray of shape (n_samples, n_reservoir + n_inputs)
            Extended states of reservoir, i.e., the X which stacks horizontally the
            inputs and the reservoir states for all times t during training, ie.
            full_states = [X(t); h(t)] for all (t)ime steps; where X is the inputs and
            h the hidden reservoir states.
            If fit_only_states is True, the shape of full_states should be
            (n_samples, n_reservoir).
        outputs: 2D np.ndarray of shape (n_samples, n_outputs)
            Target output of the training set, i.e., y(t) for all t during training.

        Returns
        -------
        W_out: 2D np.ndarray of shape (n_outputs, n_reservoir + n_inputs).
               If fit_only_states is True, shape is (n_outputs, n_reservoir + n_inputs).
            Outgoing weights matrix.
            Second dimension matches the reservoir neurons, n_outputs and bias.
        """
        if self.regression_method == "pinv":
            W_out = (np.linalg.pinv(full_states) @ outputs).T
        elif self.regression_method == "ridge":
            linreg = Ridge(
                alpha=self.ridge_alpha,
                fit_intercept=self.ridge_fit_intercept,
                normalize=self.ridge_normalize,
                max_iter=self.ridge_max_iter,
                tol=self.ridge_tol,
                solver=self.ridge_solver,
                random_state=self.random_state_,
            )
            linreg.fit(full_states, outputs, sample_weight=self.ridge_sample_weight)
            W_out = linreg.coef_
        else:
            raise ValueError("regression_method must be 'pinv' or 'ridge'")
        return W_out
