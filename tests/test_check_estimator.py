"""
Make sure the estimators are sklearn compatible
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from echoes import ESNRegressor


@parametrize_with_checks([ESNRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
