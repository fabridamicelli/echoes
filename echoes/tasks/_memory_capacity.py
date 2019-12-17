"""
MemoryCapacity class to run such a task as in defined by H. Jaeger in
"Short Term Memory in Echo State Networks" (2001).

The class wraps the EchoStateNetwork class and initializes the echo state network
instance to run the task. Thus, only the init parameters of EchoStateNetwork are
necessary.
"""
from typing import Callable, Dict

import numpy as np

from echoes.esn import EchoStateNetwork


class MemoryCapacity:
    """
    Memory capacity task as in defined by H. Jaeger
    in "Short Term Memory in Echo State Networks" (2001).

    Parameters
    ----------
    inputs_func: Callable
        Function to generate inputs. Should return a 1D np.ndarray.
        For example, np.random.uniform
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
        See EchoStateNetwork class for details.

    Attributes
    ----------
    esn_: EchoStateNetwork
        Fitted Echo State Network instance.
    forgetting_curve_: np.ndarray
        Forgetting curve MC(k) for each k in lags.
    memory_capacity_: float
        Sum over all values of the forgetting curve.
    outputs_test_: np.ndarray
        Target sequence used for testing.
        Stored for visualization (compare to prediction).
    outputs_pred_: np.ndarray
        Predicted target sequence (test).
        Stored for visualization (compare to prediction).
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
        self, inputs: np.ndarray, lags: iter, cut: int = 0
    ) -> np.ndarray:
        """
        Generate delayed versions of inputs sequence.
        One sequence is generated for each lag value.

        Parameters
        ----------
        inputs: np.ndarray
            Signal to lag. It will be flattened before lagging,
            as it is supposed to be only one input chanel.
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

    def forgetting(self, y_true, y_pred):
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
        self.esn_ = EchoStateNetwork(**self.esn_params).fit(inputs, outputs)

    def _predict(self):
        """
        Create test (inputs) data and predict using the trained model.
        Store forgetting_curve_, memory_capacity
        """
        # Test data
        inputs_test = self.inputs_func(**self.inputs_params).reshape(-1, 1)
        self.outputs_true_ = self._make_lagged_inputs(inputs_test, self.lags)
        self.outputs_pred_ = self.esn_.predict(inputs_test, mode="predictive")

        self.forgetting_curve_, self.memory_capacity_ = self.forgetting(
            self.outputs_true_, self.outputs_pred_
        )

    def run_task(self):
        """
        Run memory capacity task.
        Store results in self.forgetting_curve_, self.memory_capacity_.
        Train and test data are automatically generated.
        """
        self._fit()
        self._predict()
        return self
