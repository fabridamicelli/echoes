"""
MemoryCapacity class to run such a task as in defined by H. Jaeger in
"Short Term Memory in Echo State Networks" (2001).

The class wraps the ESNPredictive class and initializes the echo state network
instance to run the task. Thus, only the init parameters of ESNPredictive are
necessary.
"""
from typing import Callable, Dict, List, Union, Tuple

import numpy as np

from echoes import ESNPredictive


class MemoryCapacity:
    """
    Memory capacity task as in defined by H. Jaeger
    in "Short Term Memory in Echo State Networks" (2001).

    Parameters
    ----------
    inputs_func: Callable
        Function to generate inputs. Should return a 1D np.ndarray.
        For example, np.random.uniform.
        This function will receive kwargs params passed in inputs_params, so you want
        to make sure that your inputs_func receives kwargs. If it does not, you can
        simply wrap it up into one that does.
    inputs_params: Dict
        Parameters to pass to inputs_func.
        For example, {"low":-.5,
                      "high":.5,
                      "size":200}
    lags: np.ndarray
        Delays to be evaluated (memory capacity).
        For example: np.arange(1, 31, 5).
    esn_params: Dict
        Parameters to generate the Echo State Network.
        See ESNPredictive class for details.

    Attributes
    ----------
    esn_: ESNPredictive
        Fitted Echo State Network instance.
    forgetting_curve_: np.ndarray
        Forgetting curve MC(k) for each k in lags.
    memory_capacity_: float
        Sum over all values of the forgetting curve.
    forgetting_curve_train: np.ndarray
        Training forgetting curve MC(k) for each k in lags.
        This is kept for inspection, but it is not what you usually want to evaluate
        your model (ie., test performance stored in forgetting_curve_).
    memory_capacity_train: float
        Training sum over all values of the forgetting curve.
        This is kept for inspection, but it is not what you usually want to evaluate
        your model (ie., test performance stored in memory_capacity_train).
    outputs_test_: np.ndarray
        Target sequence used for testing.
        Stored for visualization (compare to prediction).
    outputs_pred_: np.ndarray
        Predicted target sequence (test).
        Stored for visualization (compare to prediction).

    Examples
    --------
    >>> n_reservoir = 20
    >>> lags = [1, 2, 5, 10, 15]

    >>> esn_params = dict(
            n_inputs=1,
            n_outputs=len(lags),    # one output neuron for each lag
            n_reservoir=n_reservoir,
            W=np.random.choice([0, .47, -.47], p=[.8, .1, .1], size=(n_reservoir, n_reservoir)),
            W_in=np.random.choice([.1, -.1], p=[.5, .5], size=(n_reservoir, 2)),
            spectral_radius=.9,
            bias=0,
            n_transient=100,
            regression_method="pinv"
        )

    >>> mc = MemoryCapacity(
            inputs_func=np.random.uniform,
            inputs_params={"low":-.5, "high":.5, "size":200},
            esn_params=esn_params,
            lags=lags
        ).fit_predict()

    >>> print(mc.forgetting_curve_)
    [0.986, 0.986, 0.975, 0.833, 0.123]
    >>> mc.memory_capacity_
    3.903
    >>>
    """

    def __init__(
        self,
        inputs_func: Callable = None,
        inputs_params: Dict = None,
        lags: np.ndarray = None,
        n_transient: int = None,
        esn_params: Dict = None,
    ) -> None:

        self.inputs_func = inputs_func
        self.inputs_params = inputs_params
        self.lags = lags
        self.esn_params = esn_params

    def _make_lagged_inputs(
        self, inputs: np.ndarray, lags: Union[List, np.ndarray], cut: int = 0
    ) -> np.ndarray:
        """
        Generate delayed versions of inputs sequence.
        One sequence is generated for each lag value.

        Parameters
        ----------
        inputs: np.ndarray
            Signal to lag. It will be flattened before lagging,
            as it is supposed to be only one input chanel.
        lags: np.ndarray
            Delays to be evaluated (memory capacity).
            For example: np.arange(1, 31, 5).
        cut: int, optional
            Number of initial steps to cut out.
            Make be at least larger than max(lags) if you want to avoid circle sequence.

        Returns
        -------
        lagged_inputs: np.ndarray of shape (len(inputs)-cut, len(lags))
            Array of lagged version of the inputs sequence.
            Each column represents U(t-k), where k is the lag.

        Examples
        --------
        >>> inputs = np.arange(5)
        >>> make_lagged_inputs(inputs, [1, 3])
        >>> array([[4., 2.],
                   [0., 3.],
                   [1., 4.],
                   [2., 0.],
                   [3., 1.]])

        >>> inputs = np.arange(5)
        >>> make_lagged_inputs(inputs, [1, 3], cut=2)
        >>> array([[1., 4.],
                   [2., 0.],
                   [3., 1.]])
        """
        assert isinstance(inputs, np.ndarray), "inputs must be np.ndarray"
        inputs = inputs.flatten()
        inputs_lagged = np.zeros((len(inputs), len(lags)))
        for col, lag in enumerate(lags):
            inputs_lagged[:, col] = np.roll(inputs, lag)
        return inputs_lagged[cut:, :]

    def forgetting(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple:
        """
        Return forgetting curve (MC(k) for all k) and Memory capacity
        (sum over all values of the curve).

        y_pred and y_true are compared column-wise,
        assuming each column contains the values for a
        give delay.
        """
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have same shape"
        # Store r_squared values
        r2s = []
        for true, pred in zip(y_true.T, y_pred.T):
            r2 = np.corrcoef(true, pred)[0, 1]
            r2s.append(0 if r2 is None else r2 ** 2)
        return r2s, np.sum(r2s)

    def _fit(self):
        """
        Create and fit echo state network according to task parameters.
        Store fitted esn_ for prediction.
        """
        # Training data
        inputs = self.inputs_func(**self.inputs_params).reshape(-1, 1)
        outputs = self._make_lagged_inputs(inputs, self.lags)
        # Instantiate estimator and fit
        self.esn_ = ESNPredictive(**self.esn_params).fit(inputs, outputs)
        self.forgetting_curve_train, self.memory_capacity_train = self.forgetting(
            outputs, self.esn_.predict(inputs)
        )

    def _predict(self):
        """
        Create test (inputs) data and predict using the trained model.
        Store forgetting_curve_, memory_capacity
        """
        # Test data
        inputs_test = self.inputs_func(**self.inputs_params).reshape(-1, 1)
        self.outputs_true_ = self._make_lagged_inputs(inputs_test, self.lags)
        self.outputs_pred_ = self.esn_.predict(inputs_test)

        self.forgetting_curve_, self.memory_capacity_ = self.forgetting(
            self.outputs_true_, self.outputs_pred_
        )

    def fit_predict(self):
        """
        Run memory capacity task.
        Prediction is made on *test* data, ie., unseen data coming from same
        distribution as for training, automatically generated.
        Store results in self.forgetting_curve_, self.memory_capacity_.
        Performance on training set is stored under self.forgetting_curve_train and
        self.memory_capacity_train.
        """
        self._fit()
        self._predict()
        return self
