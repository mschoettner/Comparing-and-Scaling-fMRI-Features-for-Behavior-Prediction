import numpy as np
import pandas as pd
import sys
import h5py
import json

from itertools import product
from scipy.linalg import norm

from src.preprocessing.sc import create_ddcm
from src.preprocessing.gsp_thomas import GSPTB_ComputeLaplacian, GSPTB_ExtractEigenmodes, GSPTB_ComputeSDI, GSPTB_ComputePSD

fmri_path = "/data/PRTNR/CHUV/RADMED/phagmann/hcp/HCP_fMRI"
# fmri_path = "/home/localadmin/hcp-behavior-prediction/data/HCP_fMRI_old"

# Parameter Combinations
n_sessions = [0.25, 0.5, 1, 2, 4]
random_states = [1,2,3,5,8,13,21,34,55,89]
fractions = [0.2,0.4,0.6,0.8,1.0]
params = list(product(n_sessions, random_states, fractions))
n, random_state, fraction = params[int(sys.argv[1])]

# Load subjects
with open(f"outputs/subjects/train_test_splits/train_test_split_rs-{random_state}_fraction-{fraction}.json") as f:
    subjects = json.load(f)


subjects_all = subjects["train"] + subjects["test"]

results_file = f"/data/PRTNR/CHUV/RADMED/phagmann/hcp/scaling_features/gsp_sessions-{n}_fraction-{fraction}_rs-{random_state}.h5"
# results_file = "/home/localadmin/hcp-behavior-prediction/outputs/precomputation_test/gsp.h5"

# Load SC matrix
A = create_ddcm(subjects["train"])
# Compute Laplacian
L = GSPTB_ComputeLaplacian(A, subtype="Norm")
# Extract eigenvectors U and eigenvalues Lambda
U, Lambda = GSPTB_ExtractEigenmodes(L)
# Find split index (half)
idx = int(U.shape[0]/2)

for sub in subjects_all:
    # Load timeseries
    timeseries = []
    for task in ["rest1", "rest2"]:
        for ses in ["ses-01", "ses-02"]:
            folder = f"{fmri_path}/{task}/derivatives/preproc/sub-{sub}/{ses}"
            file = f"{folder}/sub-{sub}_task-{task}_{ses}_desc-timeseries_scale-3.csv"
            ts = pd.read_csv(file, header=0, index_col=0).to_numpy().T
            timeseries.append(ts)
    if n <= 1:
        ts = timeseries[0]
        time_points_total = ts.shape[-1]
        time_points_shortened = int(time_points_total // (1 / n))
        ts = ts[:, :time_points_shortened]
    elif n >= 2:
        ts = np.concatenate(timeseries[:n], axis=-1)
    # Compute SDI and PSD
    SDI = GSPTB_ComputeSDI(U, ts, idx, 1)
    PSD = GSPTB_ComputePSD(U, ts, 1)
    PSD = norm(PSD, ord=2, axis=1)

    # Save to h5 file
    measures = [SDI, PSD]
    measure_names = ["SDI", "PSD"]
    for m, name in zip(measures, measure_names):
        with h5py.File(results_file, 'a') as f:
            if f"sub-{sub}" not in f:
                f.create_group(f"sub-{sub}")
            f[f"sub-{sub}"][name] = m