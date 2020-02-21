"""
Grid search for generative echo state network.
"""
from collections import namedtuple

from sklearn.model_selection import train_test_split

from ._grid_base import GridSearchBase
from echoes import ESNGenerative


class GridSearchESNGenerative(GridSearchBase):
    def _make_data(self, X, y):
        """
        Generate data for training/test for standard ESNGenerative.
        Split train/test data preserving time series order (no shuffling).

        Returns
        -------
        namedtuple: "Data", np.ndarrays
            Generative case: (y_train, y_test)
        """
        Data = namedtuple("Data", ["y_train", "y_test"])
        y_train, y_test = train_test_split(
            y, test_size=self.test_size, shuffle=False)
        return Data(y_train, y_test)

    def _evaluate_gridpoint(self, esn_params, data):
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
        esn = ESNGenerative(**esn_params).fit(None, data.y_train)
        y_pred = esn.predict(n_steps=data.y_test.shape[0])
        return self.scoring(data.y_test, y_pred)
