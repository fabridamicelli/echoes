from ._grid_base import GridSearch


class GridSearchTask(GridSearch):
    def __init__(self, task, task_params_fixed, param_grid, n_jobs=-2, verbose=5):
        """
        Task must have signature Task(**kwargs, esn_params=esn_params) and a score method.
        """
        self.task = task
        self.task_params_fixed = task_params_fixed
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
        return (self.task(**self.task_params_fixed, esn_params=esn_params)
                .score())
