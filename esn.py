"""
Echo State Network class
"""
from typing import Union, Callable, Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from utils import set_spectral_radius, identity


#TODO:make error function
## transform vectors of shape (x,) into (x,1)
## TODO: move into error (correct input shape)
#inputs = np.reshape(inputs, (len(inputs), -1)) if inputs.ndim < 2 else inputs
#outputs = np.reshape(outputs, (len(outputs), -1)) if outputs.ndim < 2 else outputs

class EchoStateNetwork:

    def __init__(self,
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
        bias=1,
        input_scaling: Union[float, np.ndarray] = None,
        teacher_forcing: bool = True,
        activation: Callable = np.tanh,
        activation_out: Callable = identity,
        regression_params: Dict = {
            "method": "pinv",
            "reg_penalty": None
            },
        n_transient: int = None,
        random_seed: int = None,
        verbose: bool = True
        ) -> None:
        """
        Parameters
        ----------
        n_inputs: int
            Number of input neurons.
        n_reservoir: int
            Number of reservoir neurons.
        n_outputs: int
            Number of output neurons.
        W: np.ndarray, optional
            Reservoir weights matrix. If None, random weights are used.
        spectral_radius: float
            Spectral radius of the reservoir weights matrix (W).
        W_in: np.ndarray, optional
            Input weights matrix by which input signal is multiplied.
            If None, random weights are used. Default None.
        W_feedb: np.ndarray, optional
            Feedback weights matrix by which teaching signal is multiplied in
            case of teaching force. Default None.
        sparsity: float, optional
            Proportion of the reservoir matrix weights forced to be zero.
            Default 0.
        noise: float, optional
            Magnitud of the noise input added to neurons at each step.
            This is used for regularization purposes and should typically be
            very small, e.g. 0.0001 or 1e-5.
        leak_rate: float, optional
            Leaking rate applied to the neurons at each step.
            Default is 1, which is no leaking. 0 would be total leakeage.
        bias: float, optional
            Value of the bias neuron, injected at each time step together with input.
            Default 1.
        input_scaling: float, np.ndarray
            NOT IMPLEMENTED
        teacher_forcing: bool, optional
            If True, the output signal gets reinjected into the reservoir
            during training.
        activation: function, optional
            Non-linear activation function applied to the neurons at each step.
            Default tanh.
        activation_out: function, optional
            NOT IMPLEMENTED. Activation function applied to the outputs.
        regression_params: Dict
            Parameters to solve the linear regression to find out outgoing weights.
            "method": str, optional
                One of ["pinv", "ridge"].
                Default pseudoinverse (pinv).
            "reg_penalty": float, optional
                Regularization penalty. Only used for Ridge regression.
        n_transient: int, optional
            Number of activity initial steps removed (not considered for training)
            in order to avoid initial instabilities.
        random_seed: int, optional
            Random seed fixed at the beginning for reproducibility of results.
        verbose: bool = True
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
        self.teacher_forcing = teacher_forcing
        self.activation = activation
        self.activation_out = activation_out  #TODO
        self.regression_params = regression_params
        self.n_transient = n_transient
        self.verbose = verbose
        if random_seed:
            np.random.seed(random_seed)
        self.init_all_weights()

    def init_incoming_weights(self):
        """Initialize random weights of matrix W_in"""
        self.W_in = np.random.rand(self.n_reservoir, self.n_inputs + 1) * 2 - 1  # +1 -> bias

    def init_reservoir_weights(self):
        """ Initialize random weights matrix of matrix W"""
        # Init random matrix centered around zero with desired spectral radius
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[np.random.rand(*W.shape) < self.sparsity] = 0
        self.W = set_spectral_radius(W, self.spectral_radius)

    def init_feedback_weights(self):
        """Initialize teacher weights W_feedb"""
        # random feedback (teacher forcing) weights:
        self.W_feedb = np.random.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    def init_all_weights(self):
        #TODO: set spectral radius for arbitrary input matrices
        if self.W is None:
            self.init_reservoir_weights()
        if self.W_in is None:
            self.init_incoming_weights()
        if self.W_feedb is None:
            self.init_feedback_weights()

    def _update_state(self, state, inputs, outputs):
        """Update reservoir states one time step"""
        if self.teacher_forcing:
            state_preac = self.W @ state + self.W_in @ inputs + self.W_feedb @ outputs
        else:
            state_preac = self.W @ state + self.W_in @ inputs
        new_state = (np.tanh(state_preac)
                     + self.noise * (np.random.rand(self.n_reservoir) - 0.5))
        # Apply leakage
        if self.leak_rate < 1:
            new_state = self.leak_rate * new_state + (1-self.leak_rate) * state

        return new_state

    def _solve_W_out(self, full_states, outputs):
        """Solve linear regression model for output weights"""
        if self.regression_params["method"] == "pinv":
            W_out = (np.linalg.pinv(full_states) @ outputs).T
        else:
        #TODO: Ridge regression
            raise NotImplementedError
        return W_out

    def fit(self, inputs, outputs, inspect=False):
        """ Fit model """
        #TODO: remove this -> function to check and throw error
        inputs = np.reshape(inputs, (len(inputs), -1)) if inputs.ndim < 2 else inputs
        outputs = np.reshape(outputs, (len(outputs), -1)) if outputs.ndim < 2 else outputs

        n_samples = inputs.shape[0]

        # Append the bias to inputs -> [1; u(t)]  #TODO: check: bias influences the states evolution?
        bias = np.ones((n_samples, 1)) * self.bias
        inputs = np.hstack((bias, inputs))
        # Collect reservoir states through the given input,output pairs
        states = np.zeros((n_samples, self.n_reservoir))
        for step in range(1, n_samples):
            states[step, :] = self._update_state(
                states[step-1], inputs[step, :], outputs[step-1, :])
        # Extend states matrix with inputs (and bias); i.e., make [1; u(t); x(t)]
        full_states = np.hstack((states, inputs))
        # Solve for W_out using full states and outputs, excluding transient
        self.W_out = self._solve_W_out(
            full_states[self.n_transient:, :], outputs[self.n_transient:, :])
        # Predict for training set
        training_prediction = full_states @ self.W_out.T

        # Keep last state for later
        self.last_state = states[-1, :]
        self.last_input = inputs[-1, :]
        self.last_output = outputs[-1, :]

        if self.verbose:
            print("training RMSE:",
                  np.sqrt(mean_squared_error(training_prediction, outputs)))

        return training_prediction

    def predict(self, inputs, generative_mode=True):
        """ """
        # Check inputs shape.TODO: move to function (utils)
        inputs = np.reshape(inputs, (len(inputs), -1)) if inputs.ndim < 2 else inputs
        n_samples = inputs.shape[0]

        # Append the bias to inputs -> [1; u(t)]
        bias = np.ones((n_samples, 1)) * self.bias
        inputs = np.hstack((bias, inputs))

        # Initialize predictions. If generative mode, begin with last state
        first_input = self.last_input if generative_mode else np.zeros(self.n_inputs)
        first_state = self.last_state if generative_mode else np.zeros(self.n_reservoir)
        first_output = self.last_output if generative_mode else np.zeros(self.n_outputs)
        print(first_input)
        inputs = np.vstack([first_input, inputs])
        states = np.vstack([first_state, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([first_output, np.zeros((n_samples, self.n_outputs))])

        # Go through samples (steps) and predict for each of them
        for step in range(1, n_samples):
            states[step, :] = self._update_state(
                states[step-1, :], inputs[step, :], outputs[step-1, :])
            full_states = np.concatenate([states[step, :], inputs[step, :]])
            outputs[step, :] = self.W_out @ full_states

        return outputs[1:]
