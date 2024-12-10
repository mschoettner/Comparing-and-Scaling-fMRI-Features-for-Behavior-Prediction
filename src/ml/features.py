"""Functions to generate the features defined on the graph that we will use for prediction.
"""

import os
import sys
import hurst
import numpy as np
import nibabel as nb

from scipy.fft import rfft, rfftfreq
from numba import njit


def compute_fALFF(fmri, freq_range=(0.01,0.08), TR=0.72):
    """Computes the fractional amplitude of low-frequency fluctations (fALFF) of a set of fMRI time-series.

    Parameters
    ----------
    fmri : array
        Can either be one subject's data or a whole dataset, with time as the last dimension.
    freq_range : tuple, optional
        Frequency range that defines the low frequencies, by default (0.01,0.08)
    TR : float, optional
        TR of the fMRI acquisition, by default 0.72

    Returns
    -------
    array
        fALFF for each region (and subject).
    """
    N = fmri.shape[-1] # number of samples
    # fast Fourier transform for each region
    fmri_f = rfft(fmri, axis=-1) # Fourier transform, axis specifies time dimension
    xf = rfftfreq(N, TR) # computes the frequencies at the center of each bin
    fmri_fa = np.abs(fmri_f) # amplitude
    # ratio of sum of amplitude across specified frequency range to sum of amplitude across whole spectrum
    cutoff_low = np.searchsorted(xf, freq_range[0], "right") # find lower index of cutoff for lower frequencies
    cutoff_high = np.searchsorted(xf, freq_range[1], "right") # find higher index of cutoff for lower frequencies
    ALFF = np.sum(fmri_fa[:,cutoff_low:cutoff_high], axis=-1) # amplitude of low frequency fluctuations
    AAFF = np.sum(fmri_fa, axis=-1) # amplitude of all frequencies
    fALFF = ALFF/AAFF # fractional ALFF
    return fALFF

def compute_hurst(fmri):
    """Computes the Hurst exponent of a rs-fMRI dataset

    Parameters
    ----------
    fmri : array
        rs-fMRI data of subjects with dimensions subjects x regions x time.

    Returns
    -------
    array
        Hurst exponents per subject and region
    """
    hurst_e_list = []
    for s in fmri:
        hurst_e_sub = []
        for ts in s:
            hurst_e_sub.append(hurst.compute_Hc(ts)[0])
        hurst_e_list.append(hurst_e_sub)
    hurst_e = np.array(hurst_e_list)
    return hurst_e

@njit()
def MSSD(x):
    """Mean square successive difference of a time-series.

    Parameters
    ----------
    x : array
        Array of time-series for which to compute MSSD.

    Returns
    -------
    float
        MSSD of the time-series.
    """
    # degrees of freedom
    df = len(x[:-1])
    # squared differences
    sqdif = []
    for i in range(df):
        sqdif.append((x[i+1]-x[i])**2)
    sqdif = np.array(sqdif)
    delta2 = np.sum(sqdif)/df
    return delta2

def compute_variability(fmri):
    """Computes the BOLD variability measured as the MSSD of each region of each subject in a resting-
    state fMRI dataset.

    Parameters
    ----------
    fmri : array
        rs-fMRI of subjects with dimensions subjects x regions x time.

    Returns
    -------
    array
        Variability of subjects x regions.
    """
    variability = []
    for sub in fmri:
        variability.append([MSSD(region) for region in sub])
    return np.array(variability)