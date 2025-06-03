"""
Make sure the estimators are sklearn compatible
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from echoes import ESNRegressor, ESNGenerator


@parametrize_with_checks([ESNRegressor()])
def test_sklearn_compatible_estimatorESNRegressor(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ESNGenerator()],
    expected_failed_checks=lambda _: {
        "check_complex_data": "Irrelevant for ESNGenerator",
        "check_dict_unchanged": "Irrelevant for ESNGenerator",
        "check_dont_overwrite_parameters": "Irrelevant for ESNGenerator",
        "check_dtype_object": "Irrelevant for ESNGenerator",
        "check_estimator_sparse_array": "Irrelevant for ESNGenerator",
        "check_estimator_sparse_matrix": "Irrelevant for ESNGenerator",
        "check_estimator_sparse_tag": "Irrelevant for ESNGenerator",
        "check_estimators_dtypes": "Irrelevant for ESNGenerator",
        "check_estimators_empty_data_messages": "Irrelevant for ESNGenerator",
        "check_estimators_fit_returns_self": "Irrelevant for ESNGenerator",
        "check_estimators_nan_inf": "Irrelevant for ESNGenerator",
        "check_estimators_overwrite_params": "Irrelevant for ESNGenerator",
        "check_estimators_pickle": "Irrelevant for ESNGenerator",
        "check_f_contiguous_array_estimator": "Irrelevant for ESNGenerator",
        "check_fit2d_1feature": "Irrelevant for ESNGenerator",
        "check_fit2d_1sample": "Irrelevant for ESNGenerator",
        "check_fit2d_predict1d": "Irrelevant for ESNGenerator",
        "check_fit_check_is_fitted": "Irrelevant for ESNGenerator",
        "check_fit_idempotent": "Irrelevant for ESNGenerator",
        "check_fit_score_takes_y": "Irrelevant for ESNGenerator",
        "check_n_features_in": "Irrelevant for ESNGenerator",
        "check_n_features_in_after_fitting": "Irrelevant for ESNGenerator",
        "check_positive_only_tag_during_fit": "Irrelevant for ESNGenerator",
        "check_readonly_memmap_input": "Irrelevant for ESNGenerator",
        "check_regressor_data_not_an_array": "Irrelevant for ESNGenerator",
        "check_regressor_multioutput": "Irrelevant for ESNGenerator",
        "check_regressors_int": "Irrelevant for ESNGenerator",
        "check_regressors_no_decision_function": "Irrelevant for ESNGenerator",
        "check_regressors_train": "Irrelevant for ESNGenerator",
    },
)
def test_sklearn_compatible_estimator_ESNGenerator(estimator, check):
    check(estimator)
