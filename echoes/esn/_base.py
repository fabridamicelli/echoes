"""
Echo State Network base class. It implements common code for predictive and generative
ESN's. It should not be instanciated, use ESNGenerative and ESNPredictive instead.
"""
from typing import Union, Callable, Dict

import numpy as np
from sklearn.linear_model import Ridge

from ._generative import ESNGenerative
from ..utils import (
    set_spectral_radius,
    identity,
    check_arrays_dimensions,
    check_model_params,
)


# TODO: scale/unscale teacher
class ESNBase:
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
        regression_params: Dict = {
            "method": "pinv",
            "solver": "lsqr",
            "fit_intercept": False,
            "regcoef": None,
        },
        n_transient: int = None,
        store_states_train: bool = False,
        store_states_pred: bool = False,
        random_seed: int = None,
    ) -> None:
        """
        Parameters
        ----------
        n_inputs: int
            Number of input neurons.
            Default None.
        n_reservoir: int
            Number of reservoir neurons.
            Default None.
        n_outputs: int
            Number of output neurons.
            Default None.
        W: np.ndarray of shape (n_reservoir, n_reservoir), optional.
            Reservoir weights matrix. If None, random weights are used (uniformly
            distributed around 0, ie., in [-0.5, 0.5).
            Be careful with the distribution of W values. Wrong W initialization
            might drastically affect test performance (even with reasonable good
            training fit).
            Default None.
        spectral_radius: float
            Spectral radius of the reservoir weights matrix (W).
        W_in: np.ndarray of shape (n_reservoir, 1+n_inputs) (1->bias), optional
            Input weights matrix by which input signal is multiplied.
            If None, random weights are used.
            Default None.
        W_feedb: np.ndarray of shape(n_reservoir, n_outputs), optional
            Feedback weights matrix by which teaching signal is multiplied in
            case of teaching force.
            Default None.
        sparsity: float, optional
            Proportion of the reservoir matrix weights forced to be zero.
            Note that with default W (centered around 0), the actual sparsity will
            be slightly more than the specified.
            Default 0.
        noise: float, optional
            Magnitud of the noise input added to neurons at each step.
            This is used for regularization purposes and should typically be
            very small, e.g. 0.0001 or 1e-5.
            Default 0.
        leak_rate: float, optional
            Leaking rate applied to the neurons at each step.
            Default is 1, which is no leaking. 0 would be total leakeage.
        bias: float, optional
            Value of the bias neuron, injected at each time step together with input.
            Default 1.
        input_scaling: float or np.ndarray of length n_inputs.
            Scalar to multiply each input before feeding it to the network.
            If float, all inputs get multiplied by same value.
            If array, it must match n_inputs length, specifying the scaling factor for
            each input.
            Default None.
        input_shift: float or np.ndarray of length n_inputs.
            Scalar to add to each input before feeding it to the network.
            If float, multiplied same value is added to all inputs.
            If array, it must match n_inputs length, specifying the value to add to
            each input.
            Default None.
        teacher_forcing: bool, optional
            If True, the output signal gets reinjected into the reservoir
            during training.
            Default False.
        activation: function, optional
            Non-linear activation function applied to the neurons at each step.
            Default tanh.
        activation_out: function, optional
            Activation function applied to the outputs. In other words, it is assumed
            that targets = f(outputs). So the output produced must be transformed.
            Default identity.
        inv_activation_out: function, optional.
            Inverse of acivation function applied to the outputs. This is used to first
            transform targets to teacher (during training).
            Default identity.
        fit_only_states: bool
            If True, outgoing weights (W_out) are computed fitting only the reservoir
            states. Inputs and bias are still use to drive reservoir activity, but
            ignored for fitting W_out, both in the training and prediction phase.
            Default False.
        regression_params: Dict
            Parameters to solve the linear regression to find out outgoing weights.
            "method": str, optional
                One of ["pinv", "ridge", "ridge_formula"]. Default pseudoinverse (pinv).
            "solver": str, optional
                Solver to use in the Ridge regression. Default least squares (lsqr).
                Valid options are the ones included in sklearn Ridge:
                   ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                Check sklearn.linear_model.Ridge documentation for details.
            "fit_intercept": bool, optional
                If True, intercept is fit in Ridge regression. Default False.
            "regcoef": float
                Regularization coefficient used for Ridge regression.
                Default is None to make sure one deliberately sets this since it is
                a crucial parameter.
                # TODO: recommend sensible range of values to try out depending on the
                task.
        n_transient: int, optional
            Number of activity initial steps removed (not considered for training)
            in order to avoid initial instabilities.
            Default is 0, but this is something one definitely might want to tweak.
            # TODO: recommend sensible range of values depending on the task.
        random_seed: int, optional
            Random seed fixed at the beginning for reproducibility of results.
            Default None.
        store_states_train: bool, optional
            If True, time series series of reservoir neurons during training are stored
            in the object attribute states_train_.
        store_states_pred: bool, optional
            If True, time series series of reservoir neurons during prediction are stored
            in the object attribute states_pred_.

        Attributes
        ----------
        W_out_ : array of shape (n_outputs, n_inputs + n_reservoir + 1)
            Outgoing weights after fitting linear regression model to predict outputs.
        training_prediction_: array of shape (n_samples, n_outputs)
            Predicted output on training data.
        states_train_: array of shape (n_samples, n_reservoir)
            If store_states_train is True, states matrix is stored for visualizing
            reservoir neurons activity during training.
            Default False.
        states_pred_: array of shape (n_samples, n_reservoir)
            If store_states_pred is True, states matrix is stored for visualizing
            reservoir neurons activity during prediction (test).
            Default False.
        """
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
        self.regression_params = regression_params
        self.n_transient = n_transient
        self.store_states_train = store_states_train
        self.store_states_pred = store_states_pred
        if random_seed:
            np.random.seed(random_seed)
        self.init_all_weights()

        check_model_params(self.__dict__)

    def init_incoming_weights(self):
        """
        Initialize random incoming weights of matrix W_in (stored in self.W_in).
        Shape (n_reservoir, n_inputs+1), where +1 corresponds to bias column.
        Note: bias and input weights are not initialized separately.
        # TODO: initialize bias weights separately, as we might want bias to have
                a different contribution than the inputs.
        """
        self.W_in = (
            np.random.rand(self.n_reservoir, self.n_inputs + 1) * 2 - 1
        )  # +1 -> bias

    def init_reservoir_weights(self):
        """
        Initialize random weights matrix of matrix W (stored in self.W).
        Shape (n_reservoir, n_reservoir).
        Spectral radius and sparsity are adjusted.
        """
        # Init random matrix centered around zero with desired spectral radius
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[np.random.rand(*W.shape) < self.sparsity] = 0
        self.W = set_spectral_radius(W, self.spectral_radius)

    def init_feedback_weights(self):
        """
        Initialize teacher feedback weights (stored inW_feedb).
        Shape (n_reservoir, n_outputs).
        """
        # random feedback (teacher forcing) weights:
        self.W_feedb = np.random.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    def init_all_weights(self):
        """
        Wrapper function to initialize all weight matrices at once.
        Even with user defined reservoir matrix W, the spectral radius is adjusted.
        """
        if self.W is None:
            self.init_reservoir_weights()
        else:
            self.W = set_spectral_radius(self.W, self.spectral_radius)
        if self.W_in is None:
            self.init_incoming_weights()
        if self.W_feedb is None and self.teacher_forcing:
            self.init_feedback_weights()

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
            np.random.rand(self.n_reservoir) - 0.5
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
        if self.regression_params["method"] == "pinv":
            W_out = (np.linalg.pinv(full_states) @ outputs).T
        elif self.regression_params["method"] == "ridge":
            linreg = Ridge(
                alpha=self.regression_params["regcoef"],
                solver=self.regression_params["solver"],
                fit_intercept=self.regression_params["fit_intercept"],
            )
            linreg.fit(full_states, outputs)
            W_out = linreg.coef_
        # TODO: test formula/write more clearly. Current one is copied from Mantas code.
        elif self.regression_params["method"] == "ridge_formula":
            Y = outputs.T
            X = full_states.T
            X_T = X.T
            I = np.eye(1 + self.n_inputs + self.n_reservoir)
            reg = self.regression_params["regcoef"]
            W_out = np.dot(
                np.dot(Y, X_T),
                np.linalg.inv(
                    np.dot(X, X_T) + reg * np.eye(1 + self.n_inputs + self.n_reservoir)
                ),
            )
        else:
            raise ValueError(
                "regression_params['method'] must be one of pinv, ridge, ridge_formula"
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

        Returns
        -------
        self: returns an instance of self.
        """
        esn_type = "generative" if isinstance(self, ESNGenerative) else "predictive"
        if esn_type == "predictive":
            assert inputs is not None, "inputs must be specified for predictive ESN"
        # If generative mode, make inputs zero, ignoring the possibly given ones
        if esn_type == "generative":
            inputs = np.zeros(shape=(outputs.shape[0], self.n_inputs))

        # Scale and shift inputs (only for predictive case)
        inputs = self.scale_shift_inputs(inputs) if esn_type == "predictive" else inputs
        # Inverse transform outputs (map them into inner, latent space)
        outputs = self.inv_activation_out(outputs)
        check_arrays_dimensions(inputs, outputs)  # sanity check of dimensions

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

        # Extend states matrix with inputs (and bias); i.e., make [1; u(t); x(t)]
        full_states = states if self.fit_only_states else np.hstack((states, inputs))

        # Solve for W_out using full states and outputs, excluding transient
        self.W_out_ = self._solve_W_out(
            full_states[self.n_transient :, :], outputs[self.n_transient :, :]
        )
        # Predict on training set (map them back to original space with activation)
        self.training_prediction_ = self.activation_out(full_states @ self.W_out_.T)

        # Keep last state for later (only generative case)
        if esn_type == "generative":
            self.last_state = states[-1, :]
            self.last_input = inputs[-1, :]
            self.last_output = outputs[-1, :]

        # Store reservoir activity
        if self.store_states_train:
            self.states_train_ = states

        return self
