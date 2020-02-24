from ._grid_base import GridSearch


class GridSearchTask(GridSearch):
    """
    Wrapper class for GridSearch on tasks.
    Task is supposed to be a class that implements a score method (used for evaluation).
    Task must have signature Task(**kwargs, esn_params=esn_params), as the task is
    internally initialized with Task(**task_params, esn_params=esn_params).

    Parameters
    ----------
    task: task class
        Task to be evaluated. Must have a score method.
    task_params: Mapping
        Parameters needed to initialize the task.
        The task is initialized with Task(**task_params, esn_params=esn_params).
    param_grid: dict of string to sequence, or sequence of dicts
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        See sklearn.model_selection.ParameterGrid for details.
    esn_params: mapping
        Parameters to instantiate esn over which the grid search is actually performed.
    n_jobs: int or None, optional, default=-2 (all processors but one)
        Number of jobs to run in parallel. See joblib library for details.
        -1 means using all processors.
        -2 means using all processors but one.
    verbose: int, default=5
        Verbosity level. See joblib library for details.
    """
    def __init__(self, task, task_params, param_grid, n_jobs=-2, verbose=5):
        self.task = task
        self.task_params = task_params
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _evaluate_gridpoint(self, esn_params, data):
        """
        Evaluate one constellation of paremeters (gridpoint).
        Instantiate Task, fit and score it.

        Parameters
        ----------
        esn_params: mapping of parameters to instantiate esn.
        data: None
           It has no effect, kept for API consistency

        Returns
        -------
        score: float
            Result of calling the score method of the ESN-like class.
        """
        return self.task(**self.task_params, esn_params=esn_params).score()
