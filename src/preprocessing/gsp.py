import numpy as np
import pygsp
import pandas as pd

from scipy.spatial.distance import euclidean

from src.preprocessing.gsp_thomas import GSPTB_ComputeLaplacian, GSPTB_ExtractEigenmodes

def distance_matrix(coords):    
    dist = np.zeros(shape=(coords.shape[0], coords.shape[0]))
    for i, rowi in enumerate(coords):
        for j, rowj in enumerate(coords):
            distance = euclidean(rowi, rowj)
            dist[i, j] = distance
    return dist

def avg_sc_matrix(sc_dict):
    # calculate average SC
    sc_array = np.array([sc_dict[subject]['SC'] for subject in sc_dict])
    mean_sc = sc_array.mean(axis=0)
    # make symmetric
    mean_sc = (mean_sc + mean_sc.T) / 2
    return mean_sc

def distance_dependent_consensus(A, dist, hemiid, nbins):
    """Generates a group-representative structural connectivity matrix, preserving the connection length distribution.
    Parameters
    ----------
    A : array
        [subject x node x node] structural connectivity matrices
    dist : array
        [node x node] distance matrix
    hemiid : array
        indicator matrix for left (0) or right (1) hemispheres
    nbins : int
        number of distance bins
    Returns
    -------
    array
        Binary group matrices with distance based consensus
    """
    # This bins the distances
    distbins = np.linspace(min(dist[dist!=0]), max(dist[dist!=0]), nbins+1)
    distbins[-1] = distbins[-1] + 1
    nsub, n, _ = A.shape # number of subjects (nsub), number of nodes (n)
    C = np.sum(A > 0, axis=0) # sum over subjects, consistency (matrix with number of subjects where connection is present)
    # W is essentially the average structural connectivity across the subjects that have a given connection
    W = np.divide(np.sum(A, axis=0), C) # element-wise right division to get average weight
    W = np.nan_to_num(W) # set nan to 0
    Grp = np.zeros(shape=(n,n,2)) # object to store inter/intra-hemispheric connections
    for j in range(2): # inter- or intra-hemispheric edge mask
        # If cross-hemispheric, we select all the connections that fit
        if j == 0:
            d = (hemiid == 0) @ (hemiid.T == 1)
            d = np.logical_or(d, d.T)
        # Else, within the same hemisphere
        else:
            d = np.logical_or((hemiid == 0) @ (hemiid.T == 0), (hemiid == 1) @ (hemiid.T == 1))
            d = np.logical_or(d, d.T) # changed it so it gives intra-hemispheric mask
        # m contains only the distance values satisfying the mask constraint, else zeros
        m = dist * d
        # D becomes a vector that contains all the distance values associated to existing connections across the pool of subjects 
        D = ((A > 0) * (dist*np.triu(d))) # SC values weighted by distance
        D = D[D!=0] # only keep non-zero values
        tgt = len(D)/nsub # Number of elements non-null per subject on average
        G = np.zeros(shape=(n,n)) # empty array with shape n by n
        G = G.flatten('F') # flatten into vector
        for ibin in range(nbins): # iterate over distance bins
            # Indices of connections that fall into the distance bin
            mask = np.nonzero(np.transpose(np.triu(np.logical_and(m >= distbins[ibin], m < distbins[ibin + 1]), 1)))
            # same indices if array was flattened
            mask_flat = np.flatnonzero(np.transpose(np.triu(np.logical_and(m >= distbins[ibin], m < distbins[ibin + 1]), 1)))
            # Per subject, how many distance values?
            frac = round(tgt * sum(np.logical_and(D >= distbins[ibin], D < distbins[ibin + 1]))/len(D))
            # For the retained connections, we sample how many subjects had it
            #c = C[mask]
            Cflat = C.flatten('F')
            c=Cflat[mask_flat]
            idx = argsort2(-c)
            G[mask_flat[idx[:frac]]] = 1
        G = np.reshape(G, (n,n),order='F')
        Grp[:,:,j] = G
    G = np.sum(Grp, axis=-1)
    G = G + G.T
    return G

def argsort2(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def compute_harmonics(con_mat, return_eigenvalues=False):
    """Compute the connectome harmonics from a given connectivity matrix

    Parameters
    ----------
    con_mat : array
        Connectivity matrix size region x region.

    Returns
    -------
    array
        Connectome harmonics size region x eigenvalue (the eigenvectors are in
        the columns)
    """
    # compute graph Laplacian/connectome harmonics
    np.fill_diagonal(con_mat, 0) # PyGSP does not support self-loops
    G_nfd = pygsp.graphs.Graph(con_mat) # create graph object from SC matrix
    G_nfd.compute_laplacian(lap_type="normalized") # compute graph laplacian
    G_nfd.compute_fourier_basis() # compute connectome harmonics
    if return_eigenvalues:
        return np.array(G_nfd.e), np.array(G_nfd.U)
    return np.array(G_nfd.U)

def extract_gsp_features(fmri_dict, sc_dict, subjects):
    # calculate average SC
    sc_array = np.array([sc_dict[subject]['SC'] for subject in sc_dict])
    mean_sc = sc_array.mean(axis=0)
    # make symmetric
    mean_sc = (mean_sc + mean_sc.T) / 2
    # calculate connectome harmonics of average SC matrix
    sc_harmonics = compute_harmonics(mean_sc)
    # calculate mean, sd of fMRI time-series
    for subject in fmri_dict:
        fmri_dict[subject]['mean'] = fmri_dict[subject]['xx'].mean(axis=1)
        fmri_dict[subject]['sd'] = fmri_dict[subject]['xx'].std(axis=1)
    # project fMRI mean and sd on connectome harmonics
    sd_harmonics = []
    mean_harmonics = []
    for subject in fmri_dict:
        sd_node = fmri_dict[subject]['sd'][16:] # signal in node-space
        mean_node = fmri_dict[subject]['mean'][16:] # signal in node-space
        sd_harmonic = sc_harmonics.T @ sd_node # signal in harmonics-space
        mean_harmonic = sc_harmonics.T @ mean_node # signal in harmonics-space
        sd_harmonics.append(sd_harmonic)
        mean_harmonics.append(mean_harmonic)
    sd_harmonics = np.array(sd_harmonics)
    mean_harmonics = np.array(mean_harmonics)
    # subset of subjects in training set
    sd_train = pd.DataFrame(sd_harmonics, index=fmri_dict.keys()).loc[subjects]
    mean_train = pd.DataFrame(mean_harmonics, index=fmri_dict.keys()).loc[subjects]

    return sd_train.to_numpy(), mean_train.to_numpy()

def extract_spectral_densities(fmri_dict, sc_dict, subjects):
    """Extracts the spectral density from the fmri signal.

    Parameters
    ----------
    sc_dict : dict
        Structural connectivity dictionary
    fmri_dict : dict
        FMRI signal dictionary

    Returns
    -------
    array
        Numpy array with the spectral densities of each subject in the rows,
        ordered by eigenvalue.
    """
    # calculate average SC
    sc_array = np.array([sc_dict[subject]['SC'] for subject in sc_dict])
    mean_sc = sc_array.mean(axis=0)
    # make symmetric
    mean_sc = (mean_sc + mean_sc.T) / 2
    # calculate connectome harmonics of average SC matrix
    sc_harmonics = compute_harmonics(mean_sc)
    spectral_densities = []
    for subject in fmri_dict:
        fmri_signal = fmri_dict[subject]["xx"][16:]
        fmri_signal_harmonics = sc_harmonics.T @ fmri_signal
        spectral_density = np.sum(abs(fmri_signal_harmonics)**2, axis=1)
        spectral_density_normalized = spectral_density/sum(spectral_density)
        spectral_densities.append(spectral_density_normalized)
    sp_den_train = pd.DataFrame(spectral_densities, index=fmri_dict.keys()).loc[subjects]
    return sp_den_train

def split_harmonics(sc_harmonics, mean_density):
    """Splits the connectome harmonics in two parts with equal energy

    Parameters
    ----------
    sc_harmonics : array
        Structural harmonics of the average SC matrix
    mean_density : array
        Spectral density averaged overs subjects

    Returns
    -------
    array, array
        Lower und higher frequency part of the connectome harmonics, respectively
    """
    # find split value using cumulative sum:
    total_energy = np.sum(mean_density)
    for i, d in enumerate(np.cumsum(mean_density)):
        if d > total_energy/2:
            split_value = i-1
            break
    # structural harmonics, renamed for convention
    U = sc_harmonics.copy()
    # split into higher and lower part by setting the other half to zero
    U_low = U.copy()
    U_low[:, split_value:] = 0
    U_high = U.copy()
    U_high[:, :split_value] = 0
    return U_low, U_high

def structural_decoupling_index(fmri_np, sc_harmonics, U_low, U_high):
    """Calculates the structural decoupling index for a dataset.

    Parameters
    ----------
    fmri_np : array
        fMRI signal for all subjects
        dimensions: subjects x regions x time points
    sc_harmonics : array
        Structural harmonics of the average SC matrix
        dimensions: regions x frequencies
    mean_density : array
        Spectral density averaged over subjects.

    Returns
    -------
    array
        Structural decoupling index for each subject and region
        dimensions: subjects x regions
    """
    # list to store SDI and FC values
    SDI_list = []
    coupled_fc_list = []
    decoupled_fc_list = []
    # structural harmonics, renamed for convention
    U = sc_harmonics.copy()
    # loop over subjects
    for i, s in enumerate(fmri_np):
        print("Calculating SDI, coupled & decoupled FC for subject", i+1)
        # list to store coupled and decoupled signal
        s_C_list = []
        s_D_list = []
        # loop over time points
        for t in s.T:
            s_C_t = U_low @ U.T @ t
            s_D_t = U_high @ U.T @ t
            s_C_list.append(s_C_t)
            s_D_list.append(s_D_t)
        # arrays: time points x regions
        s_C = np.array(s_C_list)
        s_D = np.array(s_D_list)
        # coupled and decoupled fc: region x region
        coupled_fc_sub = np.corrcoef(s_C, rowvar=False)
        decoupled_fc_sub = np.corrcoef(s_D, rowvar=False)
        # norm per region
        s_C_norm = np.linalg.norm(s_C, axis=0)
        s_D_norm = np.linalg.norm(s_D, axis=0)
        SDI_subject = s_D_norm / s_C_norm # is this element-wise?
        SDI_list.append(SDI_subject)
        coupled_fc_list.append(coupled_fc_sub)
        decoupled_fc_list.append(decoupled_fc_sub)
    # SDI array: subjects x regions
    SDI = np.array(SDI_list)
    # FC arrays: region x region
    coupled_fc = np.array(coupled_fc_list)
    decoupled_fc = np.array(decoupled_fc_list)
    return SDI, coupled_fc, decoupled_fc


def project_signal(harmonics, signal, input_type="dict"):
    if input_type=="dict":
        projected_signal = [harmonics.T @ signal[sub] for sub in signal]
    elif input_type=="array":
        projected_signal = [harmonics.T @ sub for sub in signal]
    return np.array(projected_signal)

def compute_coefficients(A, feature, n_coefficients, regions):
    # only include cortical regions
    A = pd.DataFrame(A, index=regions, columns=regions)
    A = A.filter(regex="^ctx", axis=0)
    A = A.filter(regex="^ctx", axis=1)
    # compute normalized laplacian
    L = GSPTB_ComputeLaplacian(A, subtype="Norm")
    # Extract eigenvectors U and eigenvalues Lambda
    U, Lambda = GSPTB_ExtractEigenmodes(L)
    # project signal on eigenmodes
    X = project_signal(U, feature, input_type="array")
    # take subset of coefficients
    X = X[:, :n_coefficients]
    
    return X
