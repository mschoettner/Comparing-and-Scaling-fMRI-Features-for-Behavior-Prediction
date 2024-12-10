import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from os.path import join as opj
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, cross_validate, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from scipy.stats import pearsonr


def extract_fc_vector(fc_dict, subjects):
    # lower triangle indices
    tril = np.tril_indices_from(fc_dict[100206], k=-1)

    # list of flattened FC values from the upper triangle
    fc_flat_list = []
    for subject in subjects:
        fc_flat_list.append(fc_dict[subject].to_numpy()[tril])
    # FC array with dimensions subjects x connections
    fc_vector = np.array(fc_flat_list)
    return fc_vector

def flatten_fc_array(fc_array):
    # lower triangle indices
    tril = np.tril_indices_from(fc_array[0], k=-1)

    # list of flattened FC values from the upper triangle
    fc_flat_list = []
    for fc in fc_array:
        fc_flat_list.append(fc[tril])
    # FC array with dimensions subjects x connections
    fc_vector = np.array(fc_flat_list)
    return fc_vector

def get_groups(path_restricted, subjects):
    # load restricted data
    data_res = pd.read_csv(opj(path_restricted, "hcp_behavioral_RESTRICTED.csv"), index_col=0)
    # only use subset with imaging data
    data_res = data_res.loc[subjects]
    groups = data_res['Family_ID']
    return groups

def create_training_test(subjects, path_restricted):
    # load restricted data
    data_res_all = pd.read_csv(opj(path_restricted, "hcp_behavioral_RESTRICTED.csv"), index_col=0)
    data_res = data_res_all.loc[subjects, :].copy()
    # group shuffle split
    gss_train_test = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, test_idx in gss_train_test.split(data_res, groups=data_res['Family_ID']):
        train_subjects = data_res.index[train_idx]
        test_subjects = data_res.index[test_idx]
    return train_subjects, test_subjects

def correlation(x, y):
    return pearsonr(x, y)[0]

def test_kernel_regression(X, y, groups, params, n_jobs=-1, verbose=0, return_model=False):
    # set inner and outer cv loop
    gss_cv_inner = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    gss_cv_outer = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    # set parameter space
    param_grid = {"kernelregression__alpha": params}
    # set scaler and regression
    scaler = StandardScaler()
    kr = KernelRidge(kernel="linear")
    # combine in pipeline
    pipe = Pipeline([('scaler', scaler), ('kernelregression', kr)])
    # set validation metrics
    metrics = dict(exp_var = "explained_variance", r2 = "r2", neg_mae = "neg_mean_absolute_error", r = make_scorer(correlation))

    print("Training model...")
    regressor = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=gss_cv_inner, verbose=verbose)
    regressor.fit(X, y, groups=groups)
    scores_dict = cross_validate(regressor, X, y, cv=gss_cv_outer, groups=groups, verbose=verbose, n_jobs=n_jobs, fit_params={"groups": groups}, scoring=metrics)
    scores = pd.DataFrame.from_dict(scores_dict)
    print(f"Cross-validated scores:")
    print(scores)
    print(f"Mean R2: {scores.mean()}")
    if return_model:
        return scores, regressor
    return scores

def test_kernel_regression_null(X, y, groups, params, n_jobs=-1, verbose=0, return_model=False):
    # set inner and outer cv loop
    gss_cv_inner = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    gss_cv_outer = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    # set parameter space
    param_grid = {"kernelregression__alpha": params}
    # set scaler and regression
    scaler = StandardScaler()
    kr = KernelRidge(kernel="linear")
    # combine in pipeline
    pipe = Pipeline([('scaler', scaler), ('kernelregression', kr)])
    # set validation metrics
    metrics = dict(exp_var = "explained_variance", r2 = "r2", neg_mae = "neg_mean_absolute_error", r = make_scorer(correlation))
    # reshuffle target
    y = resample(y)
    print("Training model...")
    regressor = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=gss_cv_inner, verbose=verbose)
    regressor.fit(X, y, groups=groups)
    scores_dict = cross_validate(regressor, X, y, cv=gss_cv_outer, groups=groups, verbose=verbose, n_jobs=n_jobs, fit_params={"groups": groups}, scoring=metrics)
    scores = pd.DataFrame.from_dict(scores_dict)
    print(f"Cross-validated scores:")
    print(scores)
    print(f"Mean R2: {scores.mean()}")
    if return_model:
        return scores, regressor
    return scores

def test_lasso_regression(X, y, groups, params, n_jobs=-1, verbose=0, return_model=False):
    # set inner and outer cv loop
    gss_cv_inner = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    gss_cv_outer = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    # set parameter space
    param_grid = {"lasso__alpha": params}
    # set scaler and regression
    scaler = StandardScaler()
    lasso = Lasso(max_iter=10000)
    # combine in pipeline
    pipe = Pipeline([('scaler', scaler), ('lasso', lasso)])
    # set validation metrics
    metrics = dict(exp_var = "explained_variance", r2 = "r2", neg_mae = "neg_mean_absolute_error", r = make_scorer(correlation))

    print("Training model...")
    regressor = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=gss_cv_inner, verbose=verbose)
    regressor.fit(X, y, groups=groups)
    scores_dict = cross_validate(regressor, X, y, cv=gss_cv_outer, groups=groups, verbose=verbose, n_jobs=n_jobs, fit_params={"groups": groups}, scoring=metrics)
    scores = pd.DataFrame.from_dict(scores_dict)
    print(f"Cross-validated scores:")
    print(scores)
    print(f"Mean R2: {scores.mean()}")
    if return_model:
        return scores, regressor
    return scores

def correlation(x, y):
    return pearsonr(x, y)[0]

def score_model(y, y_pred):
    scores = dict(test_r = correlation(y, y_pred),
                  test_r2 = r2_score(y, y_pred),
                  test_exp_var = explained_variance_score(y, y_pred),
                  test_neg_mae = -mean_absolute_error(y, y_pred),
                  test_neg_mse = -mean_squared_error(y, y_pred),
                  test_neg_rmse = -np.sqrt(mean_squared_error(y, y_pred)))
    scores_pd = pd.Series(scores)
    return scores_pd
