"""
Echo State Network (ESN) base class.
It implements common code for predictive and generative ESN's.
It should not be instanciated, use ESNGenerative and ESNPredictive instead.
"""
import warnings

from typing import Union, Callable, Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_random_state

from echoes.utils import (
    set_spectral_radius,
    identity,
    check_arrays_dimensions,
    check_model_params,
    check_inputs,
    check_outputs
)


# TODO: scale/unscale teacher
class ESNBase:
    """
    Parameters
    ----------
    n_inputs: int, default=None
        Number of input neurons.
    n_reservoir: int, default=None
        Number of reservoir neurons.
    n_outputs: int, default=None
        Number of output neurons.
    W: np.ndarray of shape (n_reservoir, n_reservoir), optional, default=None
        Reservoir weights matrix. If None, random weights are used (uniformly
        distributed around 0, ie., in [-0.5, 0.5).
        Be careful with the distribution of W values. Wrong W initialization
        might drastically affect test performance (even with reasonable good
        training fit).
    spectral_radius: float, default=None
        Spectral radius of the reservoir weights matrix (W).
    W_in: np.ndarray of shape (n_reservoir, 1+n_inputs) (1->bias), optional, default None.
        Input weights matrix by which input signal is multiplied.
        If None, random weights are used.
    W_feedb: np.ndarray of shape(n_reservoir, n_outputs), optional, default None.
        Feedback weights matrix by which teaching signal is multiplied in
        case of teaching force.
    sparsity: float, optional, default=0
        Proportion of the reservoir matrix weights forced to be zero.
        Note that with default W (centered around 0), the actual sparsity will
        be slightly more than the specified.
    noise: float, optional, default=0
        Magnitud of the noise input added to neurons at each step.
        This is used for regularization purposes and should typically be
        very small, e.g. 0.0001 or 1e-5.
    leak_rate: float, optional, default=1
        Leaking rate applied to the neurons at each step.
        Default is 1, which is no leaking. 0 would be total leakeage.
    bias: float, optional, default=1
        Value of the bias neuron, injected at each time step together with input.
    input_scaling: float or np.ndarray of length n_inputs, default=None
        Scalar to multiply each input before feeding it to the network.
        If float, all inputs get multiplied by same value.
        If array, it must match n_inputs length, specifying the scaling factor for
        each input.
    input_shift: float or np.ndarray of length n_inputs, default=None
        Scalar to add to each input before feeding it to the network.
        If float, multiplied same value is added to all inputs.
        If array, it must match n_inputs length, specifying the value to add to
        each input.
    teacher_forcing: bool, optional, default=False
        If True, the output signal gets reinjected into the reservoir
        during training.
    activation: function, optional, default=tanh
        Non-linear activation function applied to the neurons at each step.
    activation_out: function, optional, default=identity
        Activation function applied to the outputs. In other words, it is assumed
        that targets = f(outputs). So the output produced must be transformed.
    inv_activation_out: function, optional, default=identity
        Inverse of acivation function applied to the outputs. This is used to first
        transform targets to teacher (during training).
    fit_only_states: bool,default=False
        If True, outgoing weights (W_out) are computed fitting only the reservoir
        states. Inputs and bias are still use to drive reservoir activity, but
        ignored for fitting W_out, both in the training and prediction phase.
    regression_method: str, optional, default pseudoinverse (pinv).
        Method to solve the linear regression to find out outgoing weights.
        One of ["pinv", "ridge", "ridge_formula"].
        If "ridge" or "ridge_formula", then the ridge_* parameters will be used.
    ridge_alpha: float, ndarray of shape (n_outputs,), default=None
        Regularization coefficient used for Ridge regression.
        Larger values specify stronger regularization.
        If an array is passed, penalties are assumed to be specific to the targets.
        Hence they must correspond in number.
        Default is None to make sure one deliberately sets this since it is
        a crucial parameter. See sklearn Ridge documentation for details.
        # TODO: recommend sensible range of values depending on the task.
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
        # TODO: recommend sensible range of values depending on the task.
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

    Attributes
    ----------
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
        n_inputs: int = None,
        n_reservoir: int = None,
        n_outputs: int = None,
        W: np.ndarray = None,
        spectral_radius: float = None,
        W_in: np.ndarray = None,
        W_feedb: np.ndarray = None,
        sparsity: float = 0,
        noise: float = 0,
        leak_rate: float = 1,
        bias: Union[int, float] = 1,
        input_scaling: Union[float, np.ndarray] = None,
        input_shift: Union[float, np.ndarray] = None,
        teacher_forcing: bool = False,
        activation: Callable = np.tanh,
        activation_out: Callable = identity,
        inv_activation_out: Callable = identity,
        fit_only_states: bool = False,
        regression_method: str = "pinv",
        ridge_alpha: float = None,
        ridge_fit_intercept: bool = False,
        ridge_normalize: bool = False,
        ridge_max_iter: int = None,
        ridge_tol: float = 1e-3,
        ridge_solver: str = "auto",
        ridge_sample_weight: Union[float, np.ndarray] = None,
        n_transient: int = None,
        store_states_train: bool = False,
        store_states_pred: bool = False,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> None:

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.W = W
        self.W_in = W_in
        self.W_feedb = W_feedb
        self.sparsity = sparsity
        self.noise = noise
        self.leak_rate = leak_rate
        self.bias = bias
        self.input_scaling = input_scaling
        self.input_shift = input_shift
        self.teacher_forcing = teacher_forcing
        self.activation = activation
        self.activation_out = activation_out
        self.inv_activation_out = inv_activation_out
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

        self.init_all_weights()  # also initialize random_state_ (sklearn convention)

    def init_all_weights(self):
        """
        Wrapper function to initialize all weight matrices at once.
        Even with user defined reservoir matrix W, the spectral radius is adjusted.
        """
        # Initialize random state and store with underscore (sklearn convention)
        self.random_state_ = check_random_state(self.random_state)

        if self.W is None:
            self.init_reservoir_weights()
        else:
            self.W = set_spectral_radius(self.W, self.spectral_radius)
        if self.W_in is None:
            self.init_incoming_weights()
        if self.W_feedb is None and self.teacher_forcing:
            self.init_feedback_weights()

    def init_incoming_weights(self):
        """
        Initialize random incoming weights of matrix W_in (stored in self.W_in).
        Shape (n_reservoir, n_inputs+1), where +1 corresponds to bias column.
        Note: bias and input weights are not initialized separately.
        # TODO: initialize bias weights separately, as we might want bias to have
                a different contribution than the inputs.
        """
        self.W_in = (
            self.random_state_.rand(self.n_reservoir, self.n_inputs + 1) * 2 - 1
        )  # +1 -> bias

    def init_reservoir_weights(self):
        """
        Initialize random weights matrix of matrix W (stored in self.W).
        Shape (n_reservoir, n_reservoir).
        Spectral radius and sparsity are adjusted.
        """
        # Init random matrix centered around zero with desired spectral radius
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        self.W = set_spectral_radius(W, self.spectral_radius)

    def init_feedback_weights(self):
        """
        Initialize teacher feedback weights (stored inW_feedb).
        Shape (n_reservoir, n_outputs).
        """
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    # TODO test input scaling and shifting
    # TODO maybe move to utils
    def scale_shift_inputs(self, inputs):
        """
        Scale first and then shift inputs vector/matrix.
        """
        if self.input_scaling is not None:
            if isinstance(self.input_scaling, (float, int)):
                inputs *= self.input_scaling
            elif isinstance(self.input_scaling, np.ndarray):
                assert len(self.input_scaling) == self.n_inputs, "wrong input scaling"
                inputs *= self.input_scaling[:, None]  # broadcast column-wise
            else:
                raise ValueError("wrong input scaling type")

        if self.input_shift is not None:
            if isinstance(self.input_shift, (float, int)):
                inputs += self.input_shift
            elif isinstance(self.input_scaling, np.ndarray):
                assert len(self.input_shift) == self.n_inputs, "wrong input shift"
                inputs += self.input_scaling[:, None]  # broadcast column-wise
            else:
                raise ValueError("wrong input scaling type")

        return inputs

    def _update_state(self, state, inputs, outputs):
        """
        Update reservoir states one time step with the following equations.
        There are two cases, a) without and b) with teacher forcing (feedback):

        a.1)    x'(t) = f(W x(t-1) + W_in [1; u(t)]) + e

        a.2)    x(t) = (1-a) * x(t-1) + a * x(t)'


        b.1)    x'(t) = f(W x(t-1) + W_in [1; u(t)] + W_feedb y(t-1)) + e

        b.2)    x(t) = (1-a) * x(t-1) + a * x(t)'

        Where
            x(t): states vector at time t
            x'(t): states at time t before applying leakeage
            a: leakeage rate (1 is no leakeage, 0 is complete leakeage)
            f: activation function
            e: random noise applied to neurons (regularization)
            W: reservoir weights matrix
            W_in: incoming weights matrix
            W_feedb: feedback (teaching) matrix
            u(t): inputs vector at time t
            y(t): outputs vector at time t
            1: bias input.

        Notes
        -----
        It is asssumed that:
          - inputs already include bias.
          - states and outputs already correpond to time = t-1, while inputs
          correspond to time = t so that the code implementation corresponds
          to the described update equations. This is actually handled automa-
          tically (see fit and predict functions), so you don't have to worry
          about it.

        Returns
        -------
        states: 2D np.ndarray of shape (1, n_reservoir)
            Reservoir states vector after update.
        """
        if self.teacher_forcing:
            state_preac = self.W @ state + self.W_in @ inputs + self.W_feedb @ outputs
        else:
            state_preac = self.W @ state + self.W_in @ inputs
        new_state = self.activation(state_preac) + self.noise * (
            self.random_state_.rand(self.n_reservoir) - 0.5
        )
        # Apply leakage
        if self.leak_rate < 1:
            new_state = self.leak_rate * new_state + (1 - self.leak_rate) * state

        return new_state

    def _solve_W_out(self, full_states, outputs):
        """
        Solve for outgoing weights with linear regression, i.e., the equation:
                W_out = Y X.T inv(X X.T)

        Solution is achieved according to the parameters in regression_params.

        Parameters
        ----------
        full_states: 2D np.ndarray of shape (n_samples, n_reservoir + n_inputs + 1)
            Extended states of reservoir, i.e., the X which accumulates [x(t); 1; u(t)]
            for all times t during training. Where x are reservoir neurons states,
            1 is the bias and u are the inputs.
            If fit_only_states is True, the shape of full_states should be
            (n_samples, n_reservoir).
        outputs: 2D np.ndarray of shape (n_samples, n_outputs)
            Target output of the training set, i.e., [y(t)] for all t during training.

        Returns
        -------
        W_out: 2D np.ndarray of shape (1, n_reservoir + n_outputs + 1).
               If fit_only_states is True, shape is (1, n_reservoir + n_outputs).
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
            )
            linreg.fit(full_states, outputs, sample_weight=self.ridge_sample_weight)
            W_out = linreg.coef_
        # TODO: test formula/write more clearly. Current one is copied from Mantas code.
        elif self.regression_method == "ridge_formula":
            Y = outputs.T
            X = full_states.T
            X_T = X.T
            I = np.eye(1 + self.n_inputs + self.n_reservoir)
            reg = self.ridge_alpha
            W_out = np.dot(
                np.dot(Y, X_T),
                np.linalg.inv(
                    np.dot(X, X_T) + reg * np.eye(1 + self.n_inputs + self.n_reservoir)
                ),
            )
        else:
            raise ValueError(
                "regression_method must be one of pinv, ridge, ridge_formula"
            )
        return W_out

    def fit(self, inputs=None, outputs=None):
        """
        Fit Echo State model, i.e., find outgoing weights matrix (W_out) for later
        prediction.
        Bias is appended automatically to the inputs.

        Parameters
        ----------
        inputs: None or 2D np.ndarray of shape (n_samples, n_inputs)
            Training input, i.e., X, the features.
            If None, it is assumed that only the teaching sequence matters (outputs)
            and simply a sequence of zeros will be fed in - matching the len(outputs).
            This is to be used in the case of generative mode.
        outputs: 2D np.ndarray of shape (n_smaples, n_outputs)
            Training output, i.e., y, the target.
        esn_type: str
            Type of ESN network. Either "ESNGenerative" or "ESNPredictive"

        Returns
        -------
        self: returns an instance of self.
        """
        esn_type = self.__class__.__name__
        check_model_params(self.__dict__, esn_type)
        check_inputs(inputs, esn_type)
        check_outputs(outputs, self.n_outputs)

        # If generative mode, make inputs zero, ignoring the possibly given ones
        if esn_type == "ESNGenerative":
            inputs = np.zeros(shape=(outputs.shape[0], self.n_inputs))

        check_arrays_dimensions(inputs, outputs) # sanity check

        # Scale and shift inputs (only for predictive case)
        inputs = (
            self.scale_shift_inputs(inputs)
            if esn_type == "ESNPredictive"
            else inputs
        )
        # Inverse transform outputs (map them into inner, latent space)
        outputs = self.inv_activation_out(outputs)

        n_samples = inputs.shape[0]
        # Append the bias to inputs -> [1; u(t)]
        bias = np.ones((n_samples, 1)) * self.bias
        inputs = np.hstack((bias, inputs))
        # Collect reservoir states through the given input,output pairs
        states = np.zeros((n_samples, self.n_reservoir))
        for step in range(1, n_samples):
            states[step, :] = self._update_state(
                states[step - 1], inputs[step, :], outputs[step - 1, :]
            )

        # Extend states matrix with inputs (and bias); i.e., make [x(t); 1; u(t)]
        full_states = states if self.fit_only_states else np.hstack((states, inputs))

        # Solve for W_out using full states and outputs, excluding transient
        self.W_out_ = self._solve_W_out(
            full_states[self.n_transient :, :], outputs[self.n_transient :, :]
        )
        # Predict on training set (map them back to original space with activation)
        self.training_prediction_ = self.activation_out(full_states @ self.W_out_.T)

        # Keep last state for later (only generative case)
        if esn_type == "ESNGenerative":
            self.last_state = states[-1, :]
            self.last_input = inputs[-1, :]
            self.last_output = outputs[-1, :]

        # Store reservoir activity
        if self.store_states_train:
            self.states_train_ = states
        return self
