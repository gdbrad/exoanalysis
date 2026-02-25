import numpy as np
import gvar as gv
import lsqfit 
import scipy.linalg
import functools
import matplotlib
import matplotlib.pyplot as plt
import copy 
import itertools


"""
extract spectra in each irrep with the variational method, which solves the GEVP. 
1. jackknife on raw correlators 
2. solve GEVP on each JN sample 
3. collect eigenvalues across the samples 
4. convert to gvars 

determine energy levels in each irrep by fitting the principal correlators lambda_n(t,t0)
5. pass to lsqfit 
"""

def jackknife_blocks(data):
    """
    data: (Ncfg, ...)
    Returns jackknife means shape (Ncfg, ...)
    """

    Ncfg = data.shape[0]
    jk = []

    for i in range(Ncfg):
        mask = np.ones(Ncfg, dtype=bool)
        mask[i] = False
        jk.append(data[mask].mean(axis=0))

    return np.array(jk)


def solve_gevp(C, t0, td):
    Ct0 = C[t0]
    Ctd = C[td]

    vals, vecs = scipy.linalg.eigh(Ctd, Ct0)

    idx = np.argsort(-vals)
    return vals[idx], vecs[:, idx]


def gevp_on_sample(C_sample, t0):
    Nt = C_sample.shape[0]
    N  = C_sample.shape[1]

    lambdas = np.zeros((Nt, N))

    for t in range(Nt):
        vals, _ = solve_gevp(C_sample, t0, t)
        lambdas[t] = vals

    return lambdas


def gevp_with_jackknife(C_raw, t0):
    """
    C_raw: (Ncfg, Nt, N, N)
    """

    # 1️⃣ jackknife the RAW correlators
    C_jk = jackknife_blocks(C_raw)

    # 2️⃣ solve GEVP on each sample
    lambda_jk = np.array([
        gevp_on_sample(C_sample, t0)
        for C_sample in C_jk
    ])

    # 3️⃣ build gvar from jackknife samples
    g = gv.dataset.avg_data(lambda_jk, bstrap=False)

    return g
