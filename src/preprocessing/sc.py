"""Structural connectivity related functions."""

import joblib
import os

import pandas as pd
import numpy as np

from src.utils.load import load_coordinates
from src.preprocessing.gsp import distance_dependent_consensus, distance_matrix

def sc_dict_to_array(sc_dict):
    # create array from SC dict
    sc_list = []
    for sub in sc_dict.keys():
        try:
            sc_list.append(sc_dict[sub])
        except KeyError: # one subject or so has no SC for some reason
            continue
    sc_array = np.array(sc_list)
    return sc_array

def create_ddcm(subjects, sc_measure="normalized_fiber_density",
                sc_path="/data/PRTNR/CHUV/RADMED/phagmann/hcp/sc_measures", only_ctx=False, scale=3,
                path_coordinates = "data/lausanne_parcellation"):
    if ".txt" not in path_coordinates:
        path_coordinates = os.path.join(path_coordinates, f"lausanne2018.scale{scale}.sym.corrected_regCoords.txt")
    # load coordinates of atlas
    coords = load_coordinates(path_coordinates)
    
    # compute hemiid array
    hemiid = np.zeros(shape=(coords.shape[0],1))
    hemiid[int(coords.shape[0]/2):] = 1
    
    # compute distance matrix (euclidean distance between regions)
    dist = distance_matrix(np.array(coords))
    
    # path to SC file
    sc_path_measure = os.path.join(sc_path, f"sc_desc-{sc_measure}_scale-{scale}.joblib")
    sc_dict = joblib.load(sc_path_measure)
    
    sc_array = sc_dict_to_array(sc_dict)
    
    
    # distance-dependent consensus matrix (DDCM)
    print("Creating consensus matrix using", sc_measure)
    ddcm = distance_dependent_consensus(sc_array, dist, hemiid, 41)
    
    # simple mean
    mean_cm = np.mean(sc_array, axis=0)
    
    # weighted DDCM by multiplying with mean
    ddcm_w = ddcm * mean_cm
    
    # convert to dataframe
    ddcm_w_df = pd.DataFrame(ddcm_w, index=coords.index, columns=coords.index)
    
    if only_ctx:
        ddcm_w_df = ddcm_w_df.filter(regex="^ctx", axis=0)
        ddcm_w_df = ddcm_w_df.filter(regex="^ctx", axis=1)
    return ddcm_w_df