# This script performs nuissance regression for resting-state fMRI data. We include motion, white matter, and CSF as confounds/nuissance regressors.

# Input :  functional nifti volume, motion parameters, and additional confounds to be regressed out (e.g., white matter, and CSF)

# Output: nuissance regressed 4D nifti volume

# Usage : python Regression-fMRI.py $subject-ID $task-id (e.g., 'REST1') $task-id2 (e.g., 'rest1') $sessionID (e.g., 'ses-02'), $acquisition (e.g., 'RL') $directory-where-to-save-outputs 
# Note: make sure to change the functional directory shown below: '/home/localadmin/Documents/tmp_folder/HCP_fMRI_datalad/
# This script is particularly adapted to suit the author's bash scripts.

# Author: Anjali Tarun
# Modified by: Mikkel Schöttner

import os
from nilearn import image as nimg
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats as st
import numpy as np
import pandas as pd
import sys

# Here, we load the variables that we fed to the script when calling it
Id=sys.argv[1]
task_id=sys.argv[2]
task_id2=sys.argv[3]
sesID=sys.argv[4]
acq=sys.argv[5]
output_dir=sys.argv[6]
func_parc_dir=sys.argv[7]
masks_dir=sys.argv[8]

# This points to where we stored our data
func_dir='/data/PRTNR/CHUV/RADMED/phagmann/hcp/HCP_fMRI/'+task_id2+'/sub-'+Id+'/'+sesID+'/func/'
confounds_dir=os.path.join('/data/PRTNR/CHUV/RADMED/phagmann/hcp/HCP_fMRI/', task_id2, 'derivatives', 'HCP_preproc', f'sub-{Id}')

# This points to the specific files now
if 'rest' in task_id2:
    func = os.path.join(func_dir, f'sub-{Id}_{sesID}_task-rest_acq-{acq}_bold.nii.gz')#'rfMRI_'+task_id+'_'+acq+'_hp2000_clean.nii.gz')
else:
    func = os.path.join(func_dir, f'sub-{Id}_{sesID}_task-{task_id2}_acq-{acq}_bold.nii.gz')#'rfMRI_'+task_id+'_'+acq+'_hp2000_clean.nii.gz')
confound = os.path.join(confounds_dir, f'Movement_Regressors_{acq}.txt')
# confound_wm = os.path.join(confounds_dir, 'rfMRI_'+task_id+'_'+acq+'_WM.txt') # TODO: adjust path if necessary to fit what Thomas computed
# confound_csf = os.path.join(confounds_dir, 'rfMRI_'+task_id+'_'+acq+'_CSF.txt') # TODO: adjust path if necessary to fit what Thomas computed
mask = os.path.join(confounds_dir,f'brainmask_fs_{acq}.2.nii.gz')

# WM and CSF 
mask_WM = os.path.join(masks_dir,'mask_WM.nii')
mask_CSF = os.path.join(masks_dir,'mask_CSF.nii')

masker_WM = NiftiLabelsMasker(labels_img=mask_WM, standardize=False,
                            memory='/scratch/mschottn/nilearn_cache', verbose=5, t_r=0.72, high_pass=None, mask_img=mask)
masker_CSF = NiftiLabelsMasker(labels_img=mask_CSF, standardize=False,
                            memory='/scratch/mschottn/nilearn_cache', verbose=5, t_r=0.72, high_pass=None, mask_img=mask)

# Read motion, CSF, and WM confounds
confound_df = pd.read_fwf(confound, sep='s+', header=None)
# confound_wm_df = pd.read_fwf(confound_wm, sep='s+', header=None)
# confound_csf_df = pd.read_fwf(confound_csf, sep='s+', header=None)
confound_df.to_numpy()
# confound_wm_df.to_numpy()
# confound_csf_df.to_numpy()
confound_vars = ['X','Y','Z','RotX','RotY','RotZ','dX','dY','dZ','dRotX','dRotY','dRotZ']
# confound_df[12]=confound_wm_df
# confound_df[13]=confound_csf_df
confound_df[12] = masker_WM.fit_transform(func)
confound_df[13] = masker_CSF.fit_transform(func)

# Read imaging file
raw_func_img = nimg.load_img(func)
# Remove the first few images to achive steady state
func_img = raw_func_img.slicer[:,:,:,6:]
# Drop confound dummy TRs from the dataframe to match the size of our new func_img
drop_confound_df = confound_df.loc[6:].to_numpy()
drop_confound_df[np.isnan(drop_confound_df)] = 0

# Clean!
clean_img = nimg.clean_img(func_img,confounds=drop_confound_df,detrend=True,standardize=False, low_pass=None, high_pass=0.01, mask_img=mask, ensure_finite=True, t_r=0.72)

# Saving to nifti file
clean_dir=os.path.join(output_dir, 'sub-'+Id+'_task-'+task_id2+'_'+sesID+'_desc-preproc_bold.nii.gz')
clean_img.to_filename(clean_dir)

for i in range(5):
    func_parc = os.path.join(func_parc_dir, 'sub-'+Id+'_atlas-L2018_res-scale'+str(i+1)+'_space-MNI_dseg.nii.gz')

    # Read the parcellation
    masker = NiftiLabelsMasker(labels_img=func_parc, standardize=False,
                            memory='/scratch/mschottn/nilearn_cache', verbose=5, high_pass=None, t_r=0.72, mask_img=mask)
    
    # Read and save the parcellation image
    parc_img = nimg.load_img(func_parc)
    parc_path = os.path.join(output_dir, 'sub-'+Id+'_atlas-L2018_res-scale'+str(i+1)+'_space-MNI_dseg.nii.gz')
    parc_img.to_filename(parc_path)

    # Load the functional data (directly parcellated)
    time_series = masker.fit_transform(clean_dir)
    time_series_pd = pd.DataFrame(time_series)
    time_series_pd.to_csv(os.path.join(output_dir,'sub-'+Id+'_task-'+task_id2+'_'+sesID+'_desc-timeseries_scale-'+str(i+1)+'.csv'))

    time_series_z = st.zscore(time_series)
    time_series_z_pd = pd.DataFrame(time_series_z)
    time_series_z_pd.to_csv(os.path.join(output_dir,'sub-'+Id+'_task-'+task_id2+'_'+sesID+'_desc-timeseries_zscored_scale-'+str(i+1)+'.csv'))