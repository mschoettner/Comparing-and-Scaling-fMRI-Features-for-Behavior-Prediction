import numpy as np
import pandas as pd
import os
import sys
import hurst
import h5py

from src.ml.features import MSSD, compute_fALFF

fmri_path = "/data/PRTNR/CHUV/RADMED/phagmann/hcp/HCP_fMRI"
# fmri_path = "/home/localadmin/hcp-behavior-prediction/data/HCP_fMRI_old"

subjects = pd.read_csv('outputs/subjects/hcp_complete_rest.csv', header=None)[0].to_numpy()
# subjects = [100206, 100307, 100408, 100610]

n_sessions = [0.25, 0.5, 1, 2, 4]
n = n_sessions[int(sys.argv[1])]

results_file = f"/data/PRTNR/CHUV/RADMED/phagmann/hcp/scaling_features/fc_sessions-{n}.h5"
# results_file = "/home/localadmin/hcp-behavior-prediction/outputs/precomputation_test/simple.h5"

for sub in subjects:
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
    fc = np.corrcoef(ts)
    with h5py.File(results_file, 'a') as f:
        if f"sub-{sub}" not in f:
            f.create_group(f"sub-{sub}")
        f[f"sub-{sub}"]["FC"] = fc