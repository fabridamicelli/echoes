from typing import Dict, Callable, Union, Sequence
import warnings
from collections import namedtuple
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection._search import _check_param_grid
from sklearn.exceptions import NotFittedError

from echoes import ESNGenerative, ESNPredictive


class GridSearch:
    """
    Generic class to perform grid search over parameter grid of
    hyperparameters of Echo State Network (ESN).

    This class knows how to evaluate parameter constellations for arbitrary
    ESN-like objects (analogous to estimator in sklearn). That is, it maps
    (parameters, train_data, test_data) -> score, and stores results (along with
    best performances). In particular, train_data and test_data may be None, as the
    passed ESN-like class might generate its own data.

    The passed ESN-like class must have fit and score methods. For example, one can
    programm a task, eg memory capacity, and use the GridSearch to find best
    hyperparameters, as long as the Task class has an appropiate fit and score method.
    Note that you might want to override the _make_data method if your task requires
    special data generation.

    Higher scores are considered better.

    Parameters
    ----------
    esn: Echo State Network-like object, eg, ESNGenerative, ESNPredictive, Task
        *class* (not object as in sklearn).
        It must provide a score function with signature score(X, y) and
        return a single value.
    param_grid: dict of string to sequence, or sequence of dicts
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        See sklearn.model_selection.ParameterGrid for details.
    validation_size: int, float.
        If int, number of steps used as validation.
        If float, proportion of steps used as validation.
        Time series order is preserved, ie., validation is always the last part
        without shuffling.

    Methods
    -------
    fit: Evaluate parameter grid
    to_dataframe(self): Return results of grid search as dataframe.

    Attributes
    ----------
    params_: list of evaluated parameter constellations
    scores_: list of scores
    best_esn_: ESN-like, eg ESNPredictive or ESNGenerative
        ESN-like which gave highest score on the left out data, refitted with all
        training data.
        Available if refit True.
    best_score_: float
        Higher score over all evaluated parameter constellations.
    best_params_: dict
        Parameter constellation with best score (highest) on valid data.
    best_params_idx_: int
        Index of best parameters (to select on results_["params"]).
    """

    def __init__(
        self,
        esn: Union[ESNGenerative, ESNPredictive] = None,
        param_grid: Union[Dict, Sequence] = None,
        validation_size: Union[int, float] = None,
        refit: bool = True,
        n_jobs: int = -2,
        verbose: int = 5,
    ):

        _check_param_grid(param_grid)

        self.esn = esn
        self.param_grid = param_grid
        self.validation_size = validation_size
        self.refit = refit,
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.esn_type = self.esn.__name__

    def fit(self, X=None, y=None):
        """
        Fit and score all points of the grid.
        Set attibutes best_params_, best_score_.

        This function wraps up the grid generation (parameters) and
        the data to be passed (if any) and evaluates in parallel all the
        gridpoints.
        Functions make_grid, make_data and evaluate_gridpoint actually
        do the job. So behaviour can be changed by overloading those.


        Parameters
        ----------
        X: np.ndarray of shape (n_samples(steps), n_inputs) or None
            Training inputs. In ESNGenerative case they can be None, as they will
            be anyways ignored.
            For API consistency, pass None if X is not necessary.
        y: np.ndarray of shape (n_samples(steps), n_outputs) or None
            Training outputs (targets).
            For API consistency, pass None if y is not necessary.

        Returns
        -------
        self: instance of self fitted.
        """
        data = self._make_data(X, y)
        grid = self._make_grid()
        # Evaluate all gridpoints in parallel
        self.scores_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._evaluate_gridpoint)(gridpoint, data) for gridpoint in grid
        )

        self.params_ = list(grid)
        self.best_params_idx_ = np.argmax(self.scores_)
        self.best_params_ = self.params_[self.best_params_idx_]
        self.best_score_ = self.scores_[self.best_params_idx_]
        if hasattr(self, "refit"):  # tasks might not have this attr
            if self.refit:
                self.best_esn_ = self.esn(**self.best_params_).fit(X, y)

        return self

    def _make_grid(self):
        """Returns generator of gridpoints."""
        return ParameterGrid(self.param_grid)

    def _evaluate_gridpoint(self, esn_params, data):
        """
        Evaluate one constellation of paremeters (gridpoint).
        Instantiate echo state network (esn), fit and score it.

        esn_params: mapping of parameters to instantiate esn.
        data: namedtuple
            Data to fit and score model.
            It must include the np.ndarrays: X_train, X_test, y_train, y_test,
            even if they are not required (API consistency). So they may be just None.

        Returns
        -------
        score: float
            Result of calling the score method of the ESN-like class.
        """
        # Fit model with params and get model score
        esn = self.esn(**esn_params).fit(inputs=data.X_train, outputs=data.y_train)
        return esn.score(inputs=data.X_test, outputs=data.y_test)

    def _make_data(self, X, y):
        """
        Generate data for training/test for standard ESNPredictive.
        Split train/test data preserving time series order (no shuffling).
        If self.esn is not ESNPredictive or ESNGenerative, it still generates the
        empty data container (API consistency).

        Returns
        -------
        namedtuple: "Data", np.ndarrays
            Predictive case: (X_train, X_test, y_train, y_test)
        """
        if not hasattr(self, "esn_type"):
            return None  # API consistency

        Data = namedtuple("Data", ["X_train", "X_test", "y_train", "y_test"])
        if self.esn_type == "ESNPredictive":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.validation_size, shuffle=False)
            return Data(X_train, X_test, y_train, y_test)
        elif self.esn_type == "ESNGenerative":
            y_train, y_test = train_test_split(
                y, test_size=self.validation_size, shuffle=False)
            return Data(None, None, y_train, y_test)
        else:
            return Data(None, None, None, None)  # API consistency

    def to_dataframe(self):
        """Return results of the grid search as pandas dataframe"""
        if not hasattr(self, "scores_"):
            raise NotFittedError("GridSearch not yet fitted, call fit method first.")
        results_pd = pd.DataFrame(self.params_)
        results_pd["scores"] = self.scores_
        return results_pd
