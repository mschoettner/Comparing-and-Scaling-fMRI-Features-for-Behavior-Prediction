# This is a Python script containing all functions linked to graph signal processing (GSP)
# The following are included in different subsections:
# 1. Utilities to read and process session-wise .tsv and .csv data reflective of structural connectivity or regional activity
#
# Regarding the structural data itself
# 2. Functions related to the construction of graph shift operators (GSOs) from structural connectivity matrices
# 3. Functions related to the extraction of eigenmodes, including when focused on a specific subpart of the network (i.e., Slepians)
#
# Regarding structure/function analyses
# 4. Functions related to the treatment of functional data
# 5. Functions enabling subject fingerprinting
# 6. Functions enabling behavioral prediction
#
# Written by Thomas A.W. Bolton, Connectomics Laboratory, CHUV, Lausanne, Switzerland
# One function (distance_dependent_consensus) was kindly contributed by Mikkel Schottner

# We import standard modules
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as pp
import os
import statistics as st
from sklearn import decomposition as decomp
import pickle
import copy
import mantel
import math

##############################
# Functions enabling to read data
##############################

# Reads Jagruti-like tab files
# Path points to the file to read
# Property defines the type of values to extract
# R is the number of regions considered
def GSPTB_ReadTSV(Path,Property,R):

    # Creates a 2D array that will contain our data
    SC = np.zeros((R,R),dtype=float)

    # Reads the full file content; content contains all the lines in a list
    file = open(os.path.join(Path))
    content = file.readlines()

    # First delimits the index to use for property as a function of the user query
    property_names = content[0].split('\t')
    idx_OI = property_names.index(Property)

    # Loops and fills accordingly
    for i in range(1,len(content)):
        tmp = content[i].split('\t') 
        SC[int(tmp[0])-1,int(tmp[1])-1] = float(tmp[idx_OI])

    # Makes SC symmetrical
    SC = SC + np.transpose(SC)

    # Removes diagonal elements, set to 0 for simplicity
    for i in range(0,R):
        SC[i,i]=0

    return SC

# Reads structural connectivity data
# Pat is the path towards all .tsv files to read
# Property is the property name we want to consider
# R is an array containing the number of regions in each scale
def GSPTB_ReadSC(Path,Property,R):

    # Creation of my structural connectivity data as a dictionary with five entries (one per scale)
    SC = {}

    # Filling scale by scale...
    for i in range(0,5):
        FullPath = os.path.join(Path,'dwi','SC'+str(i+1)+'.tsv')

        # We read the data from one .tsv file per scale, and fill in the dictionary
        SC.update({'S'+str(i+1):GSPTB_ReadTSV(FullPath,Property,R[i])})
    return SC

# Reads .csv files containing activity time courses
def GSPTB_ReadCSV(Path,R):

    # Gets all the .csv file content
    file = open(Path)
    content = file.readlines()

    # We create the 2D array that will contain our data
    X = np.zeros((len(content)-1,R))

    for i in range(1,len(content)):
        tmp = content[i].split(',')

        for j in range(1,len(tmp)):
            X[i-1,j-1] = tmp[j]

    return X

# Reads .csv files containing the cognitive scores extracted from Factor Analysis
def GSPTB_ReadCSV_Behav(Path,B):

    # Gets all the .csv file content
    file = open(Path)
    content = file.readlines()

    # We create the 2D array that will contain our data, and a 1D array to store subject indices
    X = np.zeros((len(content)-1,B))
    Subj = np.zeros((len(content)-1,1))

    for i in range(1,len(content)):
        tmp = content[i].split(',')
        Subj[i-1] = tmp[0]

        for j in range(1,len(tmp)):
            X[i-1,j-1] = tmp[j]

    return X,Subj

def GSPTB_ReadTC(Path,sub,ses,task,R,is_zscored):

    # Creation of my structural connectivity data as a dictionary with five entries (one per scale)
    TC = {}

    # Filling scale by scale...
    for i in range(0,5):
        if is_zscored:
            CurrentPath = os.path.join(Path,task,ses,'sub-'+str(sub)+'_task-'+task+'_'+ses+'_desc-timeseries_zscored_scale-'+str(i+1)+'.csv')
        else:
            CurrentPath = os.path.join(Path,task,ses,'sub-'+str(sub)+'_task-'+task+'_'+ses+'_desc-timeseries_scale-'+str(i+1)+'.csv')
            
        # We read the data from one .tsv file per scale, and fill in the dictionary
        TC.update({'S'+str(i+1):GSPTB_ReadCSV(CurrentPath,R[i])})
    return TC

# We want to be able to extract data about head movement, lying in a .txt file
def GSPTB_ReadHeadMotion(Path,sub):

    # Creation of a dictionary that will contain the data across sessions, for separate head movement descriptors
    HM = {}

    # Filling scale by scale...
    for rest in [1,2]:

        Sub_dico = {}

        for ses in [1,2]:
            FullPath = os.path.join(Path,'sub-'+str(sub),'rest'+str(rest),'ses'+str(ses),'Movement_Regressors.txt')

            # Reading the .txt data
            file = open(FullPath)
            content = file.readlines()

            X = np.zeros((len(content),6))

            for i in range(1,len(content)):
                tmp = content[i].split()

                for j in range(0,6):
                    X[i,j] = float(tmp[j])    

            # Converts rotational into mm assuming 50 mm radius for the head
            X[:,3:6] = 50*(X[:,3:6]*math.pi/180)

            Sub_dico.update({ses:X})

        HM.update({rest:Sub_dico})
            
    return HM


def GSPTB_SaveData(SaveName,Data):

    f = open(SaveName+'.pkl','wb')
    pickle.dump(Data,f)
    f.close()

    test = {}
    test.update({SaveName:Data})

    sp.io.savemat(SaveName+'.mat',test)



##############################
# Functions enabling the creation of graph shift operators
##############################

def GSPTB_ComputeAverageAdjacency(PathToText,Property,R,scale):

    A_avg = np.zeros((R,R))
    S = 0

    file = open(PathToText)
    tmp = file.readlines()
    Subjects = tmp[0].split(' ')

    # Goes through all the subjects that we wish to include, all inside a .txt file
    for i in Subjects:
        # Creates the path to the appropriate .tsv file and gets the data in a 2D array
        print('Subject '+str(i)+'...')
        Path = os.path.join('/Volumes/Lorena/HCP_fMRI','sub-'+str(i),'dwi','SC'+str(scale)+'.tsv')
        A_avg = A_avg + GSPTB_ReadTSV(Path,Property,R)
        S = S + 1

    A_avg = A_avg/S

    return A_avg

def GSPTB_ComputeLaplacian(A,subtype):
    
    # Computes the diagonal degree matrix
    D = np.diag(np.sum(A,axis=1))

    # Creates the Laplacian matrix according to the desired subtype
    match subtype:
        case 'Unnorm':
            L = D - A
        case 'Norm':
            L = D - A
            Dplus = np.sqrt(np.linalg.pinv(D))
            L = np.matmul(Dplus,np.matmul(L,Dplus))
        case 'RW':
            L = D - A
            Dplus = np.sqrt(np.linalg.pinv(D))
            L = np.matmul(Dplus,L)
        case _:
            print('ERROR')

    return L

def GSPTB_ComputeModularityMatrix(A):

    # Degree vector
    d = np.sum(A,axis=1)

    # Sum of edge weights
    M = np.sum(d)/2

    # Modularity matrix
    Q = A - np.outer(d,d)/(2*M)

    return Q



##############################
# Functions enabling the creation of eigenmodes from a graph shift operator
##############################

def GSPTB_ExtractEigenmodes(GSO):

    w,v = np.linalg.eig(GSO)

    U = v

    idx = np.argsort(w)

    U_sorted = U[:,idx]
    w_sorted = w[idx]

    return U_sorted, w_sorted

# GSO must be the normalized Laplacian
def GSPTB_ExtractSlepians(GSO, focus, subnet, T):

    # Normalized variant of the adjacency, used in the computations
    A = np.eye(GSO.shape[0]) - GSO

    # Creates the matrix M, describing which elements to focus the analysis on
    M = np.diag(subnet) + focus*np.diag(1-subnet)

    # Creates the criterion used for Slepians
    #Criterion = (np.matmul(M,A)+np.matmul(A,M))/2 + (np.matmul(M,np.square(A))+np.matmul(np.square(A),M))/8 - (np.matmul(M,np.matmul(M,A)))/4
    Criterion = M - np.matmul(sp.linalg.sqrtm(GSO), np.matmul(M, sp.linalg.sqrtm(GSO)))

    # I checked that the imaginary part is essentially null, just some e-20 noise
    Criterion = np.real(Criterion)
    w,v = np.linalg.eig(Criterion)

    # Gets eigenvalues and eigenvectors (i.e., Slepians)
    khi = np.real(w)
    S = np.real(v)

    # Computes the alternative criteria as well for the eigenvectors
    mu = np.diag(np.matmul(np.transpose(S),np.matmul(M,S)))

    xi_inner = np.matmul(sp.linalg.sqrtm(GSO),np.matmul(M,sp.linalg.sqrtm(GSO)))
    xi = np.diag(np.matmul(np.transpose(S),np.matmul(xi_inner,S)))
    xi = np.real(xi)

    # Now going to reorder all variables in ascending local frequency order
    idx = np.argsort(xi)
    S_sorted = S[:,idx]
    mu_sorted = mu[idx]
    khi_sorted = khi[idx]
    xi_sorted = xi[idx]

    # Now going to remove the eigenvectors that do not satisfy the concentration constraint
    S_final = np.zeros(S_sorted.shape)
    mu_final = np.zeros(mu_sorted.shape)
    khi_final = np.zeros(khi_sorted.shape)
    xi_final = np.zeros(xi_sorted.shape)

    tmp_idx = 0

    for i in range(0,len(mu_sorted)):
        if mu_sorted[i] >= T:
            S_final[:,tmp_idx] = S_sorted[:,i]
            mu_final[tmp_idx] = mu_sorted[i]
            khi_final[tmp_idx] = khi_sorted[i]
            xi_final[tmp_idx] = xi_sorted[i]
            tmp_idx = tmp_idx + 1

    S_final = np.delete(S_final,np.arange(tmp_idx,len(mu_sorted)),axis=1)
    mu_final = np.delete(mu_final,np.arange(tmp_idx,len(mu_sorted)))
    khi_final = np.delete(khi_final,np.arange(tmp_idx,len(mu_sorted)))
    xi_final = np.delete(xi_final,np.arange(tmp_idx,len(mu_sorted)))

    return S_final,khi_final,mu_final,xi_final



##############################
# Functions enabling the treatment of functional time courses in the graph domain
##############################

# Graph Fourier transform pair
def GSPTB_MakeGFT(U,X,norma):

    if norma:
        X_hat = GSPTB_ColumnNormalize(np.matmul(np.transpose(U),X))
    else:
        X_hat = np.matmul(np.transpose(U),X)

    return X_hat

def GSPTB_MakeiGFT(U,X_hat):

    X = np.matmul(U,X_hat)

    return X

# Computation of power spectral density for a signal matrix X
def GSPTB_ComputePSD(U,X,norma):

    # Gets spectral coefficients (R x T)
    X_hat = GSPTB_MakeGFT(U,X,norma)

    # Computes the PSD (R x T)
    PSD = np.abs(X_hat)*np.abs(X_hat)

    return PSD

def GSPTB_ComputePower(U,X,norma):

    # Gets spectral coefficients (R x T)
    X_hat = GSPTB_MakeGFT(U,X,norma)

    # Computes the PSD (R x T)
    Power = np.abs(X_hat)

    return Power

# Computation of energy for a signal matrix X
def GSPTB_ComputeEnergy(U,X,Lambda,norma):

    PSD = GSPTB_ComputePSD(U,X,norma)

    # Multiplies by the eigenvalue squared
    M_int = np.power(np.abs(PSD),2)
    Energy = np.zeros((M_int.shape))

    if M_int.size == M_int.shape[0]:
        Energy = M_int*Lambda
    else:
        for t in range(0,M_int.shape[1]):
            Energy[:,t] = M_int[:,t]*Lambda

    return Energy

# We will find the median split according to four different approaches
def GSPTB_FindMedianSplit(U,X,Lambda,norma):

    # Approach 1, the dumbest: just a half-half split, thus neglecting any power information
    idx_half = int(U.shape[0]/2)

    # Computes the power spectral density for each time point
    PSD = GSPTB_ComputePSD(U,X,norma)

    idx_indiv_cumsum = np.zeros((PSD.shape[1],1))

    # For each time point, we determine the median split value
    for t in range(0,PSD.shape[1]):
        idx_indiv_cumsum[t] = GSPTB_ExtractMedianValue(PSD[:,t],Lambda)

    # Getting the mode across time points (only one value returned, the smallest)
    tmpa = sp.stats.mode(idx_indiv_cumsum,keepdims=False)

    idx_cumsum = int(tmpa.mode)

    idx_mean_cumsum = GSPTB_ExtractMedianValue(np.mean(PSD,axis=1),Lambda)

    return idx_half, idx_indiv_cumsum, idx_cumsum, idx_mean_cumsum

def GSPTB_ExtractMedianValue(x,Lambda):

    # First, we compute the cumulative sum; since x has size R by 1, A will have size R by 1 too
    A = np.cumsum(x)

    # Full value (sum over everything)
    total1 = A[-1]

    idx = -1

    for r in range(0,A.shape[0]):
        if A[r]/total1 > 0.5:
            idx = r
            break

    idx = int(idx)

    return idx


def GSPTB_ComputeSDI(U,X,idx,norma, return_ts=False):

    # Number of eigenmodes/regions
    R = X.shape[0]
    T = X.shape[1]

    X_low = np.zeros((R,T))
    X_high = np.zeros((R,T))

    # If we have one value of index per time point, we generate a new filter each time; else, just one
    if isinstance(idx,int):

        H_low = np.zeros((R,R))
        H_high = np.zeros((R,R))

        for i in range(0,R):
            if i <= int(idx):
                H_low[i,i] = 1
            else:
                H_high[i,i] = 1

        X_low = GSPTB_MakeiGFT(U,np.matmul(H_low,GSPTB_MakeGFT(U,X,norma)))
        X_high = GSPTB_MakeiGFT(U,np.matmul(H_high,GSPTB_MakeGFT(U,X,norma)))

    else:

        for t in range(0,T):

            H_low = np.zeros((R,R))
            H_high = np.zeros((R,R))

            for i in range(0,R):
                if i <= int(idx[t]):
                    H_low[i,i] = 1
                else:
                    H_high[i,i] = 1

            X_low[:,t] = GSPTB_MakeiGFT(U,np.matmul(H_low,GSPTB_ColumnNormalize(GSPTB_MakeGFT(U,X[:,t],norma))))
            X_high[:,t] = GSPTB_MakeiGFT(U,np.matmul(H_high,GSPTB_ColumnNormalize(GSPTB_MakeGFT(U,X[:,t],norma))))
    
    norm_low = sp.linalg.norm(X_low, ord=2, axis=1)
    norm_high = sp.linalg.norm(X_high, ord=2, axis=1)
    SDI = norm_high/norm_low
    # SDI = np.divide(np.abs(X_high),np.abs(X_low))
    if return_ts:
        return SDI,X_low,X_high
    else:
        return SDI

def GSPTB_ComputeAlignedComponent(U,X,n,norma):

    R = U.shape[1]

    # Creates the low-frequency filter
    H_low = np.zeros((R,R))

    for i in range(0,R):
        if i < n:
            H_low[i,i] = 1

    # Low-frequency time courses are retrieved
    X_low = GSPTB_MakeiGFT(U,np.matmul(H_low,GSPTB_MakeGFT(U,X,norma)))

    return X_low


def GSPTB_ComputeLiberalComponent(U,X,n,norma):

    R = U.shape[1]

    # Creates the low-frequency filter
    H_high = np.zeros((R,R))

    for i in range(0,R):
        if i > R - n:
            H_high[i,i] = 1

    # Low-frequency time courses are retrieved
    X_high = GSPTB_MakeiGFT(U,np.matmul(H_high,GSPTB_MakeGFT(U,X,norma)))

    return X_high


def GSPTB_ComputeCovMode(U,X,norma):

    # Goes to the spectral domain
    X_hat = GSPTB_MakeGFT(U,X,norma)

    # Computes the covariance across time and extracts a region-dimensional feature set
    Cov = np.cov(X_hat)

    w,v = np.linalg.eig(Cov)

    idx = np.argsort(w)

    v_sorted = v[:,idx]
    w_sorted = w[idx]

    Cov_mode = v_sorted[:,X_hat.shape[0]-1]

    return Cov_mode

def GSPTB_SaveTCAtTemporalities(X, Dico, sub, Property, scheme, sc, Lap_type, zscore, Norma, Feat_type, SDI_type, n_coef, rest, ses):

    GSPTB_UpdateDicoEntry_Final(Dico,np.mean(X,axis=1),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Mean',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,np.std(X,axis=1),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Std',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,GSPTB_ComputeInsta(X),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Inst',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,sp.stats.kurtosis(X,axis=1),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Kurt',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,sp.linalg.norm(X,ord=1,axis=1),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Norm1',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,sp.linalg.norm(X,ord=2,axis=1),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Norm2',rest,ses)

def GSPTB_SaveSDIAtTemporalities(X1,X2, Dico, sub, Property, scheme, sc, Lap_type, zscore, Norma, Feat_type, SDI_type, n_coef, rest, ses):

    GSPTB_UpdateDicoEntry_Final(Dico,np.divide(np.mean(X2,axis=1),np.mean(X1,axis=1)),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Mean',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,np.divide(np.std(X2,axis=1),np.std(X1,axis=1)),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Std',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,np.divide(GSPTB_ComputeInsta(X2),GSPTB_ComputeInsta(X1)),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Inst',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,np.divide(sp.stats.kurtosis(X2,axis=1),sp.stats.kurtosis(X1,axis=1)),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Kurt',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,np.divide(sp.linalg.norm(X2,ord=1,axis=1),sp.linalg.norm(X1,ord=1,axis=1)),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Norm1',rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,np.divide(sp.linalg.norm(X2,ord=2,axis=1),sp.linalg.norm(X1,ord=2,axis=1)),sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,[],[],'Norm2',rest,ses)

def GSPTB_SaveStatsAtFeatures(X, U, Lambda, Dico, sub, Property, scheme, sc, Lap_type, zscore, norma, Stats_type,rest, ses):

    GSPTB_UpdateDicoEntry_Final(Dico,GSPTB_ComputePSD(U,X,norma),sub,Property,scheme,sc,Lap_type,zscore,norma,'Stats',[],[],Stats_type,'PSD',[],rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,GSPTB_ComputePower(U,X,norma),sub,Property,scheme,sc,Lap_type,zscore,norma,'Stats',[],[],Stats_type,'Power',[],rest,ses)
    GSPTB_UpdateDicoEntry_Final(Dico,GSPTB_ComputeEnergy(U,X,Lambda,norma),sub,Property,scheme,sc,Lap_type,zscore,norma,'Stats',[],[],Stats_type,'Energy',[],rest,ses)
    
##############################
# Functions linked to the creation and update of the feature dictionary
##############################

def GSPTB_UpdateDicoEntry(Dico,NewEntry,sub,Property,scheme,sc,Lap_type,net,foc,T_mu,Feat_type,Temp_type,rest,ses):

    if Lap_type == 'Norm':
        Dico[sub][Property][scheme][sc][Lap_type][net][foc][T_mu][Feat_type][Temp_type][rest].update({ses:NewEntry})
    else:
        Dico[sub][Property][scheme][sc][Lap_type][Feat_type][Temp_type][rest].update({ses:NewEntry})

def GSPTB_GetDicoEntry(Dico,sub,Property,scheme,sc,Lap_type,net,foc,T_mu,Feat_type,Temp_type,rest,ses):

    if Lap_type == 'Norm':
        return Dico[sub][Property][scheme][sc][Lap_type][net][foc][T_mu][Feat_type][Temp_type][rest][ses]
    else:
        return Dico[sub][Property][scheme][sc][Lap_type][Feat_type][Temp_type][rest][ses]

def GSPTB_CreateDico(Subjects):

    Dico = {}
    D_ses = {1:0,2:0}
    D_rest = {1:D_ses,2:copy.deepcopy(D_ses)}
    D_temp = {'Mean':D_rest,'Std':copy.deepcopy(D_rest),'Inst':copy.deepcopy(D_rest)}
    D_feat = {'PSD':D_temp,'SDI':copy.deepcopy(D_temp)}
    D_mu = {0.5:D_feat,0.75:copy.deepcopy(D_feat)}
    D_foc = {0:D_mu,0.2:copy.deepcopy(D_mu),0.4:copy.deepcopy(D_mu),0.6:copy.deepcopy(D_mu),0.8:copy.deepcopy(D_mu),1:copy.deepcopy(D_mu)}
    D_net = {0:D_foc,1:copy.deepcopy(D_foc),2:copy.deepcopy(D_foc),3:copy.deepcopy(D_foc),4:copy.deepcopy(D_foc),5:copy.deepcopy(D_foc),6:copy.deepcopy(D_foc),7:copy.deepcopy(D_foc)}
    D_lap = {'Q':copy.deepcopy(D_feat),'RW':copy.deepcopy(D_feat),'Unnorm':copy.deepcopy(D_feat),'Norm':D_net}
    D_scale = {1:D_lap,2:copy.deepcopy(D_lap),3:copy.deepcopy(D_lap),4:copy.deepcopy(D_lap),5:copy.deepcopy(D_lap)}
    D_scheme = {'subj': D_scale,'group': copy.deepcopy(D_scale)}
    D_property = {'normalized_fiber_density':D_scheme,'number_of_fibers':copy.deepcopy(D_scheme)}
    
    Dico.update({Subjects[0]:D_property})
    
    for s in range(1,len(Subjects)):
        Dico.update({Subjects[s]:copy.deepcopy(D_property)})
    
    return Dico

def GSPTB_UpdateDicoEntry_Simplest(Dico,NewEntry,sub,Property,scheme,sc,Lap_type,Feat_type,n_coef,Temp_type,rest,ses):

    if Feat_type == 'Aligned':
        Dico[sub][Property][scheme][sc][Lap_type][Feat_type][n_coef][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'Liberal':
        Dico[sub][Property][scheme][sc][Lap_type][Feat_type][n_coef][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'Itani':
        Dico[sub][Property][scheme][sc][Lap_type][Feat_type][rest].update({ses:NewEntry})
    else:
        Dico[sub][Property][scheme][sc][Lap_type][Feat_type][Temp_type][rest].update({ses:NewEntry})

def GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,sc,Lap_type,Feat_type,n_coef,Temp_type,rest,ses):
    
    if Feat_type == 'Aligned':
        return Dico[sub][Property][scheme][sc][Lap_type][Feat_type][n_coef][Temp_type][rest][ses]
    elif Feat_type == 'Liberal':
        return Dico[sub][Property][scheme][sc][Lap_type][Feat_type][n_coef][Temp_type][rest][ses]
    elif Feat_type == 'Itani':
        return Dico[sub][Property][scheme][sc][Lap_type][Feat_type][rest][ses]
    else:
        return Dico[sub][Property][scheme][sc][Lap_type][Feat_type][Temp_type][rest][ses]

def GSPTB_CreateDico_Simplest(Subjects):

    Dico = {}
    D_ses = {1:0,2:0}
    D_rest = {1:D_ses,2:copy.deepcopy(D_ses)}
    D_temp = {'Mean':D_rest,'Std':copy.deepcopy(D_rest),'Inst':copy.deepcopy(D_rest)}
    
    D_n_coef = {}
    D_n_coef.update({5:D_temp})
    for n_coef in range(10,45,5):
        D_n_coef.update({n_coef:copy.deepcopy(D_temp)})

    D_feat = {'PSD':D_temp,'SDI':copy.deepcopy(D_temp),'Aligned':D_n_coef,'Liberal':copy.deepcopy(D_n_coef),'Itani':copy.deepcopy(D_rest)}
    D_lap = {'Q':copy.deepcopy(D_feat),'RW':copy.deepcopy(D_feat),'Unnorm':copy.deepcopy(D_feat),'Norm':copy.deepcopy(D_feat)}
    D_scale = {1:D_lap,2:copy.deepcopy(D_lap),3:copy.deepcopy(D_lap),4:copy.deepcopy(D_lap),5:copy.deepcopy(D_lap)}
    D_scheme = {'subj': D_scale,'group': copy.deepcopy(D_scale)}
    D_property = {'normalized_fiber_density':D_scheme,'number_of_fibers':copy.deepcopy(D_scheme),'FA_median':copy.deepcopy(D_scheme),'fiber_length_median':copy.deepcopy(D_scheme)}
    
    Dico.update({Subjects[0]:D_property})
    
    for s in range(1,len(Subjects)):
        Dico.update({Subjects[s]:copy.deepcopy(D_property)})
    
    return Dico

def GSPTB_UpdateDicoEntry_Final(Dico,NewEntry,sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,Stats_type,Stats_type2,Temp_type,rest,ses):

    if Feat_type == 'Aligned':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][n_coef][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'Liberal':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][n_coef][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'CovMode':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][rest].update({ses:NewEntry})
    elif Feat_type == 'SDI':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][SDI_type][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'Power':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'PSD':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'Energy':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Temp_type][rest].update({ses:NewEntry})
    elif Feat_type == 'Stats':
        Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Stats_type][Stats_type2][rest].update({ses:NewEntry})

def GSPTB_GetDicoEntry_Final(Dico,sub,Property,scheme,sc,Lap_type,zscore,Norma,Feat_type,SDI_type,n_coef,Stats_type,Stats_type2,Temp_type,rest,ses):
    
    if Feat_type == 'Aligned':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][n_coef][Temp_type][rest][ses]
    elif Feat_type == 'Liberal':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][n_coef][Temp_type][rest][ses]
    elif Feat_type == 'CovMode':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][rest][ses]
    elif Feat_type == 'SDI':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][SDI_type][Temp_type][rest][ses]
    elif Feat_type == 'Power':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Temp_type][rest][ses]
    elif Feat_type == 'PSD':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Temp_type][rest][ses]
    elif Feat_type == 'Energy':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Temp_type][rest][ses]
    elif Feat_type == 'Stats':
        return Dico[sub][Property][scheme][sc][Lap_type][zscore][Norma][Feat_type][Stats_type][Stats_type2][rest][ses]

def GSPTB_CreateDico_Final(Subjects):

    Dico = {}

    # Two sessions (LR and RL)
    D_ses = {1:0,2:0}

    # Two days where scanning was performed
    D_rest = {1:D_ses,2:copy.deepcopy(D_ses)}

    # Six potential ways to aggregate time
    D_temp = {'Mean':D_rest,'Std':copy.deepcopy(D_rest),'Inst':copy.deepcopy(D_rest), 'Kurt':copy.deepcopy(D_rest), 'Norm1':copy.deepcopy(D_rest), 'Norm2':copy.deepcopy(D_rest)}
    
    # For alignment/liberality, we have different parameter choices for the number of harmonics taken together
    D_n_coef = {0:D_temp}

    for i in range(1,10):
        D_n_coef.update({i:copy.deepcopy(D_temp)})

    D_stats2 = {'PSD':D_rest,'Power':copy.deepcopy(D_rest),'Energy':copy.deepcopy(D_rest)}
    D_stats = {'Mean':D_stats2,'Std':copy.deepcopy(D_stats2),'Inst':copy.deepcopy(D_stats2), 'Kurt':copy.deepcopy(D_stats2), 'Norm1':copy.deepcopy(D_stats2), 'Norm2':copy.deepcopy(D_stats2)}
    
    D_SDI = {'Half': D_temp, 'Indiv_cumsum': copy.deepcopy(D_temp), 'Indiv_trapz': copy.deepcopy(D_temp), 'Mode_cumsum': copy.deepcopy(D_temp), 'Mode_trapz': copy.deepcopy(D_temp), 'Mean_cumsum': copy.deepcopy(D_temp), 'Mean_trapz': copy.deepcopy(D_temp)}

    # Types of features that can be computed: PSD, power, energy, SDI, alignment, liberality, covariance or statistics
    D_feat = {'PSD':D_temp,'Power':copy.deepcopy(D_temp), 'Energy':copy.deepcopy(D_temp), 'SDI':copy.deepcopy(D_SDI),'Aligned':D_n_coef,'Liberal':copy.deepcopy(D_n_coef),'CovMode':copy.deepcopy(D_rest), 'Stats':copy.deepcopy(D_stats)}
    
    # Do we normalize the time courses per column or not?
    D_norma = {True: D_feat, False: copy.deepcopy(D_feat)}

    # Do we temporally z-score the data or not?
    D_zscored = {True: D_norma, False: copy.deepcopy(D_norma)}

    # Which Laplacian do we consider?
    D_lap = {'Q':D_zscored,'RW':copy.deepcopy(D_zscored),'Unnorm':copy.deepcopy(D_zscored),'Norm':copy.deepcopy(D_zscored)}

    # At which scale are we analyzing our data (1 to 5)?
    D_scale = {1:D_lap,2:copy.deepcopy(D_lap),3:copy.deepcopy(D_lap),4:copy.deepcopy(D_lap),5:copy.deepcopy(D_lap)}

    # Which type of scheme do we use to compute the graph (one per subject or one per group)?
    D_scheme = {'subj': D_scale,'group': copy.deepcopy(D_scale)}

    # Which properties do we consider to create our graph?
    D_property = {'normalized_fiber_density':D_scheme,'number_of_fibers':copy.deepcopy(D_scheme),'FA_median':copy.deepcopy(D_scheme),'fiber_length_median':copy.deepcopy(D_scheme)}
    
    Dico.update({Subjects[0]:D_property})
    
    for s in range(1,len(Subjects)):
        Dico.update({Subjects[s]:copy.deepcopy(D_property)})
    
    return Dico



##############################
# "Edgy" functions
##############################

# Computes the edge-wise co-fluctuation time series from regional time courses
def GSPTB_ComputeCoFluctuations(X,to_keep):

    T = X.shape[1]
    R = X.shape[0]

    # Z-scores across time
    X2 = sp.stats.zscore(X,axis=1)

    Y = np.zeros((np.sum(to_keep),T))

    # Product per time point
    for t in range(0,T):

        print(t)

        tmp = X2[:,t]
        Mat = np.outer(tmp,tmp)

        idx = 0
        idx2 = 0

        for r1 in range(0,R):
            for r2 in range(0,R):
                if r2 > r1:
                    if to_keep[idx] == 1:
                        Y[idx2,t] = Mat[r1,r2]
                        idx2 += 1
                    idx += 1

    return Y

# Creates an edge-wise graph representation from a set of structural graph across subjects
# A is of size R x R x S
def GSPTB_CreateEdgyGraph(A,idx_toexclude):

    R = A.shape[0]
    S = A.shape[2]
    E = int(R*(R-1)/2)

    # First, we vectorize the matrices
    EdgyMatrix = np.zeros((int(R*(R-1)/2),S))
    idx_vec = np.zeros((int(R*(R-1)/2),1))
    
    for s in range(0,S):
        
        idx = 0

        for r1 in range(0,R):
            for r2 in range(0,R):
                if r2 > r1:
                    EdgyMatrix[idx,s] = A[r1,r2,s]
                    idx += 1

    idx = 0

    for r1 in range(0,R):
        for r2 in range(0,R):
            if r2 > r1:
                idx_vec[idx] = idx_toexclude[r1,r2]
                idx += 1

    A_meta = np.zeros((E,E))

    for e1 in range(0,E):

        print(e1)

        if idx_vec[e1] == 0:
            for e2 in range(0,E):
                if idx_vec[e2] == 0:
                    
                    if e1 == e2:
                        A_meta[e1,e2] = 0
                    elif e1 > e2:
                        
                        x1 = EdgyMatrix[e1,:]
                        x2 = EdgyMatrix[e2,:]

                        A_meta[e1,e2] = np.corrcoef(x1,x2)[0,1]

    # Gets a full matrix
    A_meta = A_meta + np.transpose(A_meta)

    # Now we want to remove the useless information
    rows_tokeep = (np.sum(A_meta,axis=1)!=0)

    A_int = np.zeros((np.sum(rows_tokeep),A_meta.shape[1]))
    A_final = np.zeros((np.sum(rows_tokeep),np.sum(rows_tokeep)))

    id = 0

    for a in range(0,len(rows_tokeep)):

        if rows_tokeep[a] == 1:
            A_int[id,:] = A_meta[a,:]
            id += 1

    id = 0

    for b in range(0,len(rows_tokeep)):

        if rows_tokeep[b] == 1:
            A_final[:,id] = A_int[:,b]
            id += 1

    return A_final, rows_tokeep


##############################
# Other miscellaneous functions
##############################

def GSPTB_ComputeInsta(X):

    T = X.shape[1]
    R = X.shape[0]
    
    Out = np.sum(np.abs(np.diff(X,axis=1)),axis=1)/(T-1)

    return Out

def GSPTB_ColumnNormalize(X):

    R = X.shape[0]

    if X.size == X.shape[0]:
        T = 1

        Out = (X - np.mean(X))
        Out = Out/np.linalg.norm(Out,ord=2)

    else:
        T = X.shape[1]
        Out = np.zeros((R,T))

        for t in range(0,T):
            tmp = X[:,t]
            Out[:,t] = (tmp - np.mean(tmp))
            Out[:,t] = Out[:,t]/np.linalg.norm(Out[:,t],ord=2)

    return Out

# This function computes a similarity matrix with a defined distance and defined feature data
# The data can come from one session or from two different sessions; in the former case, F2 should be set to an empty vector
def GSPTB_ComputeSimilarityMatrix(F1,F2, distance):

    # Number of subjects and feature values
    if F1.shape[0] == F1.size:
        S = F1.size

        # Will contain our similarity values
        Sim = np.zeros((S,S))

        for s1 in range(0,S):
            for s2 in range(0,S):
                Sim[s1,s2] = GSPTB_GetEuclideanDistance(F1[s1],F2[s2])
    else:

        S = F1.shape[1]
        F = F1.shape[0]

        # Will contain our similarity values
        Sim = np.zeros((S,S))

        # We loop across all pairs of subjects and compute the distance
        for s1 in range(0,S):
            for s2 in range(0,S):
                match distance:
                    case 'Euclidean':
                        Sim[s1,s2] = GSPTB_GetEuclideanDistance(F1[:,s1],F2[:,s2])
                    case 'Correlation':
                        Sim[s1,s2] = GSPTB_GetCorrelationDistance(F1[:,s1],F2[:,s2])
                    case _:
                        Sim[s1,s2] = -1337
            
    return Sim
            
def GSPTB_GetEuclideanDistance(x,y):

    return np.sum(np.sqrt(np.square(x-y)))

def GSPTB_GetCorrelationDistance(x,y):

    tmp = np.corrcoef(x,y)

    return 1 - tmp[0,1]

# Gets telling fingerprinting metrics from a cross-session similarity matrix
def GSPTB_GetFingerprintingMetrics(M):

    # Number of subjects
    S = M.shape[0]

    # All diagonal and off-diagonal coefficients to be extracted; note that
    # for off-diagonal ones, we want to sample both the upper and lower diagonals
    # as we consider a non-symmetrical matrix
    Diag = np.zeros((S,1))
    OffDiag = np.zeros((int(S*(S-1)),1))

    OffDiag_Summary = np.zeros((21,1))
    SubjectWise_Summary = np.zeros((S,3))

    idx_diag = 0
    idx_offdiag = 0

    for i in range(0,S):
        for j in range(0,S):
            if i == j:
                Diag[idx_diag] = M[i,j]
                idx_diag += 1
            else:
                OffDiag[idx_offdiag] = M[i,j]
                idx_offdiag += 1

        # Gets all except the diagonal element for the subject at hand
        tmp = M[i,:]
        boo = np.ones(len(tmp),dtype=bool)
        boo[i] = False
        tmp = tmp[boo]

        SubjectWise_Summary[i,1] = np.mean(tmp)
        SubjectWise_Summary[i,2] = np.std(tmp)

    # For off-diagonal elements, we look at percentiles
    OffDiag_Summary = np.percentile(OffDiag,np.linspace(0,100,21))
    SubjectWise_Summary[:,0] = Diag[:,0]
    
    return OffDiag_Summary,SubjectWise_Summary

# This function performs a Mantel test between the two 2D arrays provided as input
# at a given significance threshold (alpha level, default = 0.05)
def GSPTB_MantelTest(M1,M2,n_null):
    
    res = mantel.test(M1,M2,perms=n_null,method='spearman',tail='two-tail')

    return res.p


def GSPTB_Dis2Sim(dis):

    sim = - (dis - 1)

    return sim


def GSPTB_FillDataMatrices(R,S,Dico,sub,sub_idx,Property,idx_prop,scheme,idx_scheme,scale_oi,Lap_type,idx_lap,Feat_type,idx_metric,coef,Temp_type,Features_all,SimMat_IntraSession,SimMat_InterSession):
    
    FEATURES_11 = np.zeros((R,S))
    FEATURES_12 = np.zeros((R,S))
    FEATURES_21 = np.zeros((R,S))
    FEATURES_22 = np.zeros((R,S))

    FEATURES_11[:,sub_idx] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,1,1)
    FEATURES_12[:,sub_idx] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,1,2)
    FEATURES_21[:,sub_idx] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,2,1)
    FEATURES_22[:,sub_idx] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,2,2)
    
    # We also save that same data in Features_all, which will contain all our features across all subcases
    Features_all[:,sub_idx,idx_prop,idx_scheme,idx_lap,idx_metric,0,0] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,1,1)
    Features_all[:,sub_idx,idx_prop,idx_scheme,idx_lap,idx_metric,0,1] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,1,2)
    Features_all[:,sub_idx,idx_prop,idx_scheme,idx_lap,idx_metric,1,0] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,2,1)
    Features_all[:,sub_idx,idx_prop,idx_scheme,idx_lap,idx_metric,1,1] = GSPTB_GetDicoEntry_Simplest(Dico,sub,Property,scheme,scale_oi,Lap_type,Feat_type,coef,Temp_type,2,2)

    # Adds the data to the SimMat arrays
    SimMat_IntraSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,0] = GSPTB_ComputeSimilarityMatrix(FEATURES_11,FEATURES_11,'Correlation')
    SimMat_IntraSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,1] = GSPTB_ComputeSimilarityMatrix(FEATURES_12,FEATURES_12,'Correlation')
    SimMat_IntraSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,2] = GSPTB_ComputeSimilarityMatrix(FEATURES_21,FEATURES_21,'Correlation')
    SimMat_IntraSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,3] = GSPTB_ComputeSimilarityMatrix(FEATURES_22,FEATURES_22,'Correlation')

    SimMat_InterSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,0] = GSPTB_ComputeSimilarityMatrix(FEATURES_11,FEATURES_12,'Correlation')
    SimMat_InterSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,1] = GSPTB_ComputeSimilarityMatrix(FEATURES_11,FEATURES_21,'Correlation')
    SimMat_InterSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,2] = GSPTB_ComputeSimilarityMatrix(FEATURES_11,FEATURES_22,'Correlation')
    SimMat_InterSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,3] = GSPTB_ComputeSimilarityMatrix(FEATURES_12,FEATURES_21,'Correlation')
    SimMat_InterSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,4] = GSPTB_ComputeSimilarityMatrix(FEATURES_12,FEATURES_22,'Correlation')
    SimMat_InterSession[:,:,idx_prop,idx_scheme,idx_lap,idx_metric,5] = GSPTB_ComputeSimilarityMatrix(FEATURES_21,FEATURES_22,'Correlation')

    return Features_all, SimMat_IntraSession, SimMat_InterSession


def GSPTB_RemoveSC(A,idx):

    if A.shape[0] == A.shape[1]:
        A2 = np.zeros((int(np.sum(idx)),A.shape[0]))
        A3 = np.zeros((int(np.sum(idx)),int(np.sum(idx))))

        idx_a2 = 0

        for i in range(0,A.shape[0]):
            if idx[i]:
                A2[idx_a2,:] = A[i,:]

                idx_a2 += 1

        idx_a2 = 0

        for i in range(0,A.shape[0]):
            if idx[i]:        
                A3[:,idx_a2] = A2[:,i]

                idx_a2 += 1    
    else:
        A3 = np.zeros((int(np.sum(idx)),A.shape[1]))

        idx_a3 = 0

        for i in range(0,A.shape[0]):
            if idx[i]:
                A3[idx_a3,:] = A[i,:]

                idx_a3 += 1

    return A3