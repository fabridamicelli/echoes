from collections import namedtuple

from sklearn.model_selection import train_test_split

from ._grid_base import GridSearchBase
from echoes import ESNPredictive

# TODO: remove
class GridSearchESNPredictive(GridSearchBase):
    def _make_data(self, X, y):
        """
        Generate data for training/test for standard ESNPredictive.
        Split train/test data preserving time series order (no shuffling).

        Returns
        -------
        namedtuple: "Data", np.ndarrays
            Predictive case: (X_train, X_test, y_train, y_test)
        """
        Data = namedtuple("Data", ["X_train", "X_test", "y_train", "y_test"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.validation_size, shuffle=False)
        return Data(X_train, X_test, y_train, y_test)

    def ___evaluate_gridpoint(self, esn_params, data):
        """
        Evaluate one constellation of paremeters (gridpoint).
        Instantiate echo state network (esn), fit and score it.

        esn_params: mapping of parameters to instantiate esn.
        data: namedtuple
            It must include the np.ndarrays: X_train  X_test, y_train, y_test

        Returns
        -------
        score: float
            Result of evaluation scoring(y_test, y_pred)
        """
        esn = ESNPredictive(**esn_params).fit(data.X_train, data.y_train)
        y_pred = esn.predict(data.X_test)

        if self.strip_transient:
            n_transient = esn_params["n_transient"]
            y_true, y_pred = data.y_test[n_transient:, :], y_pred[n_transient:, :]
        else:
            y_true, y_pred = data.y_test, y_pred

        return self.scoring(y_true, y_pred)
