import os
from os.path import join as opj

import numpy as np
import pandas as pd
import scipy.io as sio
import networkx as nx

from scipy.linalg import norm


def load_dwi(dwi_path, scale=3, subs_dwi=None, verbose=False):
    """Loads the connectivity data saved in .tsv format.

    Parameters
    ----------
    folder_dwi : str
        Folder where connectivity data is stored
    scale : int, optional
        Scale of the Lausanne atlas, by default 3

    Returns
    -------
    dict
        Connectivity dictionary
    """
    if subs_dwi == None:
        subs_dwi = os.listdir(dwi_path) # list of subject folders
        subs_dwi = [s[4:] for s in subs_dwi] # only get the ID of each subject
        subs_dwi.sort() # sort
    dwi_dict = {} # dictionary to store SCs in
    filename_dwi = f"_atlas-L2018_res-scale{scale}_conndata-network_connectivity.tsv" # naming scheme
    for sub in subs_dwi: # iterating over subjects
        if verbose:
            print(f"Loading subject {sub}")
        dwi_dict[int(sub)] = pd.read_csv(opj(dwi_path, f"sub-{sub}", "dwi", f"sub-{sub}{filename_dwi}"), sep="\t")
    return dwi_dict


def create_graphs(dwi_dict, measure="normalized_fiber_density"):
    """Creates a dictionary of networkx graphs from a dictionary of connectivity data.

    Parameters
    ----------
    dwi_dict : dict
        Dictionary of connectivity data.
    measure : str, optional
        Connectivity measure, by default "normalized_fiber_density"

    Returns
    -------
    dict
        Dictionary of graphs
    """
    graph_dict = {} # dictionary to store graphs in
    for sub in dwi_dict: # iterate over subjects
        print("Creating graph for subject", sub)
        G = nx.from_pandas_edgelist(dwi_dict[sub], source="source", target="target",
                                    edge_attr=measure, create_using=nx.Graph)
        G.remove_edges_from(nx.selfloop_edges(G))
        graph_dict[sub] = G
    return graph_dict

def create_sc(dwi_path, scale=3, measure="normalized_fiber_density", subs_dwi=None, verbose=False):
    """Creates the SC matrices from the structural connectivity data for a given measure and scale

    Parameters
    ----------
    dwi_path : str
        path to DWI data
    scale : int, optional
        Scale of Lausanne atlas, by default 3
    measure : str, optional
        Measure to use as edge weight, by default "normalized_fiber_density"
    subs_dwi : list or None, optional
        List of subjects to use, if None, uses all in the given folder, by default None

    Returns
    -------
    dict
        Dictionary with SC matrices
    """
    dwi_dict = load_dwi(dwi_path, scale=scale, subs_dwi=subs_dwi)
    graph_dict = create_graphs(dwi_dict, measure=measure)
    n_nodes = [126, 172, 274, 504, 1060]
    
    sc_dict = {}
    for sub in graph_dict:
        if verbose:
            print("Extracting SC matrix for subject", sub)
        G = graph_dict[sub]
        nodes = list(G.nodes)
        nodes.sort() # sort nodes so that the order is preserved
        sc = nx.to_numpy_array(G, weight=measure, nodelist=nodes)
        if sc.shape != (n_nodes[scale-1], n_nodes[scale-1]):
            print(f"Different number of nodes detected for subject {sub}, skipping...")
            continue
        sc_dict[sub] = sc
    return sc_dict

def create_sc_manually(dwi_path, scale=3, measure="normalized_fiber_density", subs_dwi=None):
    """Creates the SC matrices from the structural connectivity data for a given measure and scale,
    but without using networkx.

    Parameters
    ----------
    dwi_path : str
        path to DWI data
    scale : int, optional
        Scale of Lausanne atlas, by default 3
    measure : str, optional
        Measure to use as edge weight, by default "normalized_fiber_density"
    subs_dwi : list or None, optional
        List of subjects to use, if None, uses all in the given folder, by default None

    Returns
    -------
    dict
        Dictionary with SC matrices
    """
    dwi_dict = load_dwi(dwi_path, scale=scale, subs_dwi=subs_dwi)
    n_nodes = [126, 172, 274, 504, 1060] # number of nodes per scale
    n = n_nodes[scale-1] # choose scale
    adjacencies = {}
    for sub in dwi_dict: # iterate over subjects
        A = np.zeros(shape=(n, n)) # create empty adjacency with n x n nodes
        for index, row in dwi_dict[sub].iterrows(): # iterate over connections
            A[int(row["source"]-1), int(row["target"]-1)] = row[measure]
        A += A.T # fill lower diagonal
        np.fill_diagonal(A, 0)
        adjacencies[sub] = A
    return adjacencies

def load_fmri_measures(path_measures, name_measure="BOLD-variability", task="rest1", session="ses-01", scale=3,
                       verbose=False, subjects="all", return_as_array=False):
    """Loads the fMRI measures data

    Parameters
    ----------
    path_measures : str
        Path to the data
    name : str
        Name of the measure to use
    task : str, optional
        Which task to use, by default "rest1"
    session : str, optional
        Which session to use, by default "ses-01"
    scale : int, optional
        Scale of Lausanne atlas, by default 3
    verbose : bool, optional
        Verbosity, by default False

    Returns
    -------
    dict
        Dictionary of fMRI measure
    """
    if subjects == "all":
        subs_fmri = os.listdir(path_measures) # subject list from folder names
        subs_fmri = [int(s[4:]) for s in subs_fmri]
        subs_fmri.sort()
    else:
        subs_fmri = subjects
    measure = {} # dictionary to store measure
    for sub in subs_fmri:
        filepath = opj(path_measures, "sub-"+str(sub), session, f"sub-{sub}_task-{task}_{session}_desc-{name_measure}_scale-{scale}.npy")
        try:
            m = np.load(filepath) # load numpy object
            if m.shape != (274,): # sometimes the file has not all nodes, skip these
                if verbose:
                    print("Subject", sub, "has different shape:", m.shape, " skipping...")
                continue
            measure[sub] = m
        except FileNotFoundError: # check if file exists
            if verbose:
                print("No file found for subject", sub)
            continue
    if not return_as_array:
        return measure
    return np.array(list(measure.values()))

def load_coordinates(path_coords):
    coords_colors = pd.read_csv(path_coords,
                            skipinitialspace=True,sep="\s*[,]\s*", engine="python") # get rid of trailing spaces
    coords_colors = coords_colors[~coords_colors["Structures Names"].str.contains("cer-")] # we do not have the cerebellum
    coords = pd.DataFrame(data=np.array([coords_colors["XCoord(mm)"], coords_colors["YCoord(mm)"], coords_colors["ZCoord(mm)"]]).T,
                          columns=["x", "y", "z"], index=list(coords_colors["Structures Names"]))
    return coords

def get_groups(path_restricted, subjects):
    # load restricted data
    data_res = pd.read_csv(opj(path_restricted, "hcp_behavioral_RESTRICTED.csv"), index_col=0)
    # only use subset with imaging data
    data_res = data_res.loc[subjects]
    groups = data_res['Family_ID']
    return groups

def load_sdi(path, subjects, task="rest1", ses="ses-01", laplacian="normalized", idx_type="cumsum", spectral_normalized=True, return_as_array=True):
    """Loads the SDI as an array of subjects x regions

    Parameters
    ----------
    path : str
        Path to the measures of that task
    subjects : str
        Subject list
    task : str, optional
        Task, by default "rest1"
    ses : str, optional
        Session, by default "ses-01"
    laplacian : str, optional
        Type of laplacian, can be "normalized" or "modularity matrix", by default "normalized"
    idx_type : str, optional
        Type of index to find cutoff, can be "cumsum", "half", "indiv_cumsum", "mean_cumsum", by default "cumsum"
    spectral_normalized : bool, optional
        Whether to use the spectral-normalized version, by default True

    Returns
    -------
    array
        Numpy array of SDI with dimensions subjects x regions
    """
    # list to store individual SDI arrays
    SDI = []
    if spectral_normalized:
        # keyword for if spectral normalized
        sn = "_spectral-normalized"
    else:
        sn = ""
    for sub in subjects:
        # load individual SDI arrays
        path_sub = os.path.join(path, f"sub-{sub}", ses, f"sub-{sub}_task-{task}_{ses}_desc-SDI_scale-3{sn}_laplacian-{laplacian}_idx-{idx_type}.npy")
        sdi = np.load(path_sub)
        SDI.append(sdi)
    if return_as_array:
        return np.array(SDI)
    else:
        return pd.DataFrame(np.array(SDI), index=subjects)

def load_psd(path, subjects, task="rest1", ses="ses-01", laplacian="normalized", spectral_normalized=True, return_as_array=True):
    """Loads the PSD as an array of subjects x coefficients

    Parameters
    ----------
    path : str
        Path to the measures of that task
    subjects : str
        Subject list
    task : str, optional
        Task, by default "rest1"
    ses : str, optional
        Session, by default "ses-01"
    laplacian : str, optional
        Type of laplacian, can be "normalized" or "modularity matrix", by default "normalized"
    spectral_normalized : bool, optional
        Whether to use the spectral-normalized version, by default True

    Returns
    -------
    array
        # Numpy array of PSD with dimensions subjects x coefficients
        List of numpy arrays with dimensions coefficients x timepoints
    """
    # list to store individual PSD arrays
    PSD = []
    if spectral_normalized:
        # keyword for if spectral normalized
        sn = "_spectral-normalized"
    else:
        sn = ""
    for sub in subjects:
        # load individual PSD arrays
        path_sub = os.path.join(path, f"sub-{sub}", ses, f"sub-{sub}_task-{task}_{ses}_desc-PSD_scale-3{sn}_laplacian-{laplacian}.npy")
        psd = norm(np.load(path_sub), ord=2, axis=1)
        PSD.append(psd)
    if return_as_array:
        return np.array(PSD)
    else:
        return pd.DataFrame(np.array(PSD), index=subjects)

def load_fc(path, subjects, task, ses, z_scored=True, scale=3, vectorized=True):
    if z_scored:
        z = "_zscored"
    else:
        z = ""
    FC = []
    for sub in subjects:
        sub_path = opj(path, task, "derivatives", "measures_old", f"sub-{sub}", ses)
        sub_file = opj(sub_path, f"sub-{sub}_task-{task}_{ses}_desc-fc{z}_scale-{scale}.npy")
        fc = np.load(sub_file)
        if vectorized:
            indices = np.tril_indices(fc.shape[0], -1)
            fc = fc[indices]
        FC.append(fc)

    return np.array(FC)
    