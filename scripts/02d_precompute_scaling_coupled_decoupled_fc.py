import numpy as np
import pandas as pd
import sys
import h5py
import json

from itertools import product
from scipy.linalg import norm

from src.preprocessing.sc import create_ddcm
from src.preprocessing.gsp_thomas import GSPTB_ComputeLaplacian, GSPTB_ExtractEigenmodes, GSPTB_ComputeSDI, GSPTB_ComputePSD, GSPTB_MakeiGFT, GSPTB_MakeGFT
from src.preprocessing.gsp import project_signal

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

results_file = f"/data/PRTNR/CHUV/RADMED/phagmann/hcp/scaling_features/coupled-decoupled-fc_sessions-{n}_fraction-{fraction}_rs-{random_state}.h5"
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
    # Compute coupled and decoupled FC
    
    # Number of eigenmodes/regions
    R = ts.shape[0]
    T = ts.shape[1]
    H_low = np.zeros((R,R))
    H_high = np.zeros((R,R))

    for i in range(0,R):
        if i <= int(idx):
            H_low[i,i] = 1
        else:
            H_high[i,i] = 1
    ts_low = GSPTB_MakeiGFT(U,np.matmul(H_low,GSPTB_MakeGFT(U,ts,1)))
    ts_high = GSPTB_MakeiGFT(U,np.matmul(H_high,GSPTB_MakeGFT(U,ts,1)))
    fc_coupled = np.corrcoef(ts_low)
    fc_decoupled = np.corrcoef(ts_high)

    # Save to h5 file
    measures = [fc_coupled, fc_decoupled]
    measure_names = ["fc_coupled", "fc_decoupled"]
    for m, name in zip(measures, measure_names):
        with h5py.File(results_file, 'a') as f:
            if f"sub-{sub}" not in f:
                f.create_group(f"sub-{sub}")
            f[f"sub-{sub}"][name] = m