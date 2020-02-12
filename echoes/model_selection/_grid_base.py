from typing import Dict, Callable, Union
import warnings

import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection._search import _check_param_grid
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
import pandas as pd


class GridSearchBase:
    """
    Generic class to perform grid search over parameter grid of
    hyperparameters of Echo State Network (ESN).

    Do not instantiate this class, but rather the children classes
    according to the case.
    For predictive mode, use GridSearchPredictiveESN.
    For generative mode, use GridSearchGenerativeESN.

    This class knows how to map the scoriong function over data and gridpoits
    (parameter constellations) independently of the specific evaluation definition.
    Subclasses implement the actual point evaluation. So the user might generate
    arbitrary grid searches by overloading the methods _make_data and _eval_point.
    For example, you can wrap up an arbitrary task under the methods _make_data and
    _eval_point in order to find best hyperparameters for the task.
    _eval_point must return a single value - the score.

    Note: best_estimator_ is considered to be the one with *lowest score*, as it is
    supposed to be a loss function.

    Parameters
    ----------
    param_grid: dict of string to sequence, or sequence of dicts
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        See sklearn.model_selection.ParameterGrid for details.
    test_size: int, float.
        If int, number of steps used as test.
        If float, proportion of steps used as test.
        Time series order is preserved, ie., test is always the last part
        without shuffling.
    scoring: callable, optional, default=mean_squared_error
        *Loss function* with signature loss(y_true, y_pred), must return a single value.
        The estimator (esn) that *minimizes* this function will be stored as best_estimator_.
    strip_transient: bool, optional, default=False
        If True, the first n_transient steps are removed before evaluation.
        This is typically what you want for ESNPredictive but NOT for ESNGenerative.

    Methods
    -------
    to_dataframe(self): Return results of grid search as dataframe.

    Attributes
    ----------
    params_: list of evaluated parameter constellations
    scores_: list of scores (loss)
    best_estimator_: ESNPredictive or ESNGenerative
        Estimator which gave highest score on the left out data.
        Available if refit True.
    best_score_: float
        Lowest loss over all evaluated parameter constellations.
    best_params_: dict
        Parameter constellation with best score (lowest loss) on valid data.
    best_params_idx_: int
        Index of best parameters (to select on results_["params"]).
    """

    def __init__(
        self,
        param_grid: Dict = None,
        test_size: Union[int, float] = None,
        scoring: Callable = mean_squared_error,
        strip_transient: bool = False,
        refit: bool = True,
        n_jobs: int = -2,
        verbose: int = 5,
    ):

        _check_param_grid(param_grid)

        self.param_grid = param_grid
        self.test_size = test_size
        self.scoring = scoring
        self.strip_transient = strip_transient
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.esn_type = (
            "predictive"
            if self.__class__.__name__ == "GridSearchESNPredictive"
            else "generative"
        )

        if self.esn_type == "predictive" and not self.strip_transient:
            warnings.warn(
                "Initial transient is being considered for the loss, which "
                "overestimates it. You might want to set strip_transient=True"
            )

        if self.esn_type == "generative" and self.strip_transient:
            raise ValueError("strip_transient must be False for generative esn.")

    def fit(self, X, y):
        """
        Fit and score all points of the grid.

        This function wraps up the grid generation (parameters) and
        the data to be passed (if any) and evaluates in parallel all the
        gridpoints.
        Functions make_grid, make_data and evaluate_gridpoint actually
        do the job. So behaviour can be changed by overloading those.

        Set attibutes best_params_, best_score_.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples(steps), n_inputs)
            Training inputs. In ESNGenerative case they can be None, as they will
            be anyways ignored.
        y: np.ndarray of shape (n_samples(steps), n_outputs)
            Training outputs (targets).

        Returns
        -------
        self: instance of self fitted.
        """
        if X is None:
            assert self.esn_type == "generative", "only ESNGenerative allows X=None"

        data = self._make_data(X, y)
        grid = self._make_grid()
        # Evaluate all gridpoints in parallel
        self.scores_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._eval_gridpoint)(gridpoint, data) for gridpoint in grid
        )

        self.params_ = list(grid)
        self.best_params_idx_ = np.argmin(self.scores_)
        self.best_params_ = self.params_[self.best_params_idx_]
        self.best_score_ = self.scores_[self.best_params_idx_]

        return self

    def _make_grid(self):
        """Returns generator of gridpoints."""
        return ParameterGrid(self.param_grid)

    def _make_data(self, X, y):
        """Implemented in the subclasses"""
        raise NotImplementedError

    def _evaluate_gridpoint(self, *args):
        """Implemented in the subclasses"""
        raise NotImplementedError

    def to_dataframe(self):
        """Return results of the grid search as pandas dataframe"""
        if not hasattr(self, "scores_"):
            raise NotFittedError("GridSearch not yet fitted, call fit method first.")
        results_pd = pd.DataFrame(self.params_)
        results_pd["scores"] = self.scores_
        return results_pd
