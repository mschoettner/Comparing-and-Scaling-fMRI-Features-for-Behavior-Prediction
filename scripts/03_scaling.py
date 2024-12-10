import h5py
import pandas as pd
import numpy as np
import sys
import json
import os
import time

from itertools import product
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA, PCA
from sklearn.kernel_ridge import KernelRidge

from src.utils.load import get_groups
from src.ml.prediction import score_model

# timer
time1 = time.time()

path_results = "/data/PRTNR/CHUV/RADMED/phagmann/hcp/scaling_features"
# path_results = "/home/localadmin/hcp-behavior-prediction/data/scaling_measures_test"
path_scores = "/data/PRTNR/CHUV/RADMED/phagmann/hcp/HCP_behavioral/hcp_pheno.csv"
# path_scores = "/home/localadmin/hcp-behavior-prediction/data/hcp_pheno.csv"
path_restricted = "/data/PRTNR/CHUV/RADMED/phagmann/hcp/HCP_behavioral"
# path_restricted = "/home/localadmin/hcp-behavior-prediction/data/HCP_behavioral_datalad/data/01_inputs"
path_out = "outputs/scaling"

# Parameter Combinations
n_sessions = [0.25, 0.5, 1, 2, 4]
fractions = [0.2,0.4,0.6,0.8,1.0]
random_states = [1,2,3,5,8,13,21,34,55,89]
params = list(product(n_sessions, random_states, fractions))
n, random_state, fraction = params[int(sys.argv[1])]
# n, random_state, fraction = 0.25, 1, 1.0

print(f"Sessions: {n}, Random state: {random_state}, Fraction: {fraction}")

# Load subjects
with open(f"outputs/subjects/train_test_splits/train_test_split_rs-{random_state}_fraction-{fraction}.json") as f:
    subjects = json.load(f)

# Features
features = ["mean", "std", "variability", "fALFF", "hurst", "FC", "SDI", "PSD",
            "gcn_1dconv", "gcn_average", "gcn_max", "gcn_r2", "gcn_std",
            "fc_coupled", "fc_decoupled"]
# features = ["fc_coupled", "fc_decoupled"]
# features = ["FC"]
# features = ["gcn_1dconv", "gcn_average", "gcn_max","gcn_r2", "gcn_std"]
# Load features
features_dict = {}
for feature in features:
    if feature == "FC":
        filepath = path_results + f"/fc_sessions-{n}.h5"
    elif feature == "fc_coupled" or feature == "fc_decoupled":
        filepath = path_results + f"/coupled-decoupled-fc_sessions-{n}_fraction-{fraction}_rs-{random_state}.h5"
    elif feature == "SDI" or feature == "PSD":
        filepath = path_results + f"/gsp_sessions-{n}_fraction-{fraction}_rs-{random_state}.h5"
    elif "gcn" in feature:
        filepath = path_results + f"/gcn_sessions-{n}_fraction-{fraction}_rs-{random_state}.h5"
    else:
        filepath = path_results + f"/simple_sessions-{n}.h5"
    with h5py.File(filepath, 'r') as f:
        feature_dict = {}
        for sub in subjects["train"] + subjects["test"]:
            feature_sub = f[f"sub-{sub}"][feature][()]
            if feature == "FC" or feature == "fc_coupled" or feature == "fc_decoupled":
                # Only upper triangular part of FC matrix
                feature_sub = feature_sub[np.triu_indices(feature_sub.shape[0], k=1)]
            feature_dict[sub] = feature_sub
                
        features_dict[feature] = pd.DataFrame.from_dict(feature_dict, orient='index')

# Load scores
scores_all = pd.read_csv(path_scores, index_col=0)
scores = scores_all.loc[subjects["train"] + subjects["test"]]

# Center and scale age
scores["age"] = (scores["age"] - scores["age"].mean()) / scores["age"].std()

# Get groups
groups_train = get_groups(path_restricted, subjects["train"])

# Set parameter ranges
alpha = np.logspace(-4, 4, 20)
l1_ratio = np.linspace(0.1, 1.0, 10)
c_param = np.logspace(-4, 2, 20)

# Cross-validation
gss_cv_inner = GroupKFold(n_splits=3)

# Model
elastic_net = ElasticNet()
kernel_ridge = KernelRidge(kernel="linear")
scaler = StandardScaler()
elnet_classifier = SGDClassifier(loss="log_loss", penalty="elasticnet")
svm = LinearSVC()
pca = PCA(n_components=min(274, round(len(subjects["train"])*2/3-15)))
pca_all = PCA()

# Suppress convergence warning
import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# Adjust feature list for FC variations
if "FC" in features:
    features.remove("FC")
    features.extend(["FC_pca", "FC_pca_all", "FC_KR"])

# Object to store results
cv_scores = []
cv_scores_gender = []

for feature in features:
    for target in scores.columns:
        if feature == "FC_pca":
            pipeline = Pipeline([
                ("scaler", scaler),
                ("pca", pca)
            ])
            if target == "gender":
                pipeline.steps.append(("model", elnet_classifier))
            else:
                pipeline.steps.append(("model", elastic_net))
            param_grid = {"model__alpha": alpha, "model__l1_ratio": l1_ratio}
            feature_data = features_dict["FC"]
        elif feature == "FC_pca_all":
            pipeline = Pipeline([
                ("scaler", scaler),
                ("pca", pca_all)
            ])
            if target == "gender":
                pipeline.steps.append(("model", svm))
                param_grid = {"model__C": c_param}
            else:
                pipeline.steps.append(("model", kernel_ridge))
                param_grid = {"model__alpha": alpha}
            feature_data = features_dict["FC"]
        elif feature == "FC_KR":
            pipeline = Pipeline([
                ("scaler", scaler)
            ])
            if target == "gender":
                pipeline.steps.append(("model", svm))
                param_grid = {"model__C": c_param}
            else:
                pipeline.steps.append(("model", kernel_ridge))
                param_grid = {"model__alpha": alpha}
            feature_data = features_dict["FC"]
        elif  feature == "fc_coupled" or feature == "fc_decoupled":
            pipeline = Pipeline([
                ("scaler", scaler)
            ])
            if target == "gender":
                pipeline.steps.append(("model", svm))
                param_grid = {"model__C": c_param}
            else:
                pipeline.steps.append(("model", kernel_ridge))
                param_grid = {"model__alpha": alpha}
            feature_data = features_dict[feature]
        else:
            pipeline = Pipeline([
                ("scaler", scaler)
            ])
            if target == "gender":
                pipeline.steps.append(("model", elnet_classifier))
            else:
                pipeline.steps.append(("model", elastic_net))
            param_grid = {"model__alpha": alpha, "model__l1_ratio": l1_ratio}
            feature_data = features_dict[feature]
        print(f"Feature: {feature}, Target: {target}")
        
        # Grid search
        grid_search = GridSearchCV(pipeline, param_grid, cv=gss_cv_inner, n_jobs=-1, verbose=1)
        X_train = feature_data.loc[subjects["train"]]
        X_test = feature_data.loc[subjects["test"]]
        y_train = scores.loc[subjects["train"], target]
        y_test = scores.loc[subjects["test"], target]
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        # Predict and score
        if target == "gender":
            accuracy = grid_search.score(X_test, y_test)
            # Save scores
            name = f"feature-{feature}_target-{target}_rs-{random_state}_fraction-{fraction}_sessions-{n}"
            row = [name, feature, target, random_state, fraction, n, accuracy]
            cv_scores_gender.append(row)

        else:
            y_pred = grid_search.predict(X_test)
            scores_dict = score_model(y_test, y_pred)
            pred_scores = list(pd.Series(scores_dict))
            print(scores_dict)

            # Save scores
            name = f"feature-{feature}_target-{target}_rs-{random_state}_fraction-{fraction}_sessions-{n}"
            row = [name, feature, target, random_state, fraction, n]
            row.extend(pred_scores)
            cv_scores.append(row)

# Save results
cv_columns = ["name", "feature", "target", "random_state", "fraction", "sessions"]
cv_columns.extend(list(scores_dict.keys()))
cv_columns_gender = ["name", "feature", "target", "random_state", "fraction", "sessions", "accuracy"]

cv_scores_df = pd.DataFrame(cv_scores, columns=cv_columns)
cv_scores_gender_df = pd.DataFrame(cv_scores_gender, columns=cv_columns_gender)
if not os.path.exists(path_out):
    os.makedirs(path_out)
cv_scores_df.to_csv(f"{path_out}/cv_scores_rs-{random_state}_fraction-{fraction}_sessions-{n}.csv")
cv_scores_gender_df.to_csv(f"{path_out}/cv_scores_rs-{random_state}_fraction-{fraction}_sessions-{n}_gender.csv")

time2 = time.time()
timespan = time2-time1
print("Runtime:")
print(timespan, "seconds")
print(time.strftime("%H:%M:%S", time.gmtime(timespan)))