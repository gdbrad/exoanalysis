import numpy as np 
import gvar as gv 
from scipy import linalg as la

def bootstrap_to_gvar(Cboot):
    """
    Convert bootstrap samples to gvar array.
    Cboot shape: (Nboot, Lt)
    """
    mean = np.mean(Cboot, axis=0)
    cov  = np.cov(Cboot.reshape(Cboot.shape[0], -1),rowvar=False)
    return gv.gvar(mean, cov)

def jack_to_gvar(Cjk):
    """
    NOT USED BC OF GV.DATASET.AVG_DATA
    Cjk shape: (Ncfg, Lt)
    """
    N = Cjk.shape[0]
    mean = np.mean(Cjk, axis=0)
    flat = Cjk.reshape(N, -1)
    cov = np.cov(flat, rowvar=False) * (N - 1)
    return gv.gvar(mean, cov)


def effective_mass(C):
    return gv.log(C[:-1] / C[1:])

def solve_gevp_bootstrap(Cboot, t0):
    """
    Cboot shape: (Nboot, Lt, N, N)
    Returns:
        lam_boot shape (Nboot, Lt, N)
    """
    Nboot, Lt, N, _ = Cboot.shape
    lam_boot = np.zeros((Nboot, Lt, N))

    for b in range(Nboot):
        Cb = Cboot[b]
        C0 = Cb[t0]
#        C0inv = np.linalg.inv(C0)
        C0inv = la.eigh(C0)

        for t in range(Lt):
            M = C0inv @ Cb[t]
            w, _ = la.eigh(M)
            idx = np.argsort(w)[::-1]
            lam_boot[b, t] = w[idx].real
    return lam_boot


def solve_gevp_jack(Cjk, t0, reg=1e-10):
    """
    cholesky reduction uses mean as metric
    """
    Ncfg, Lt, N, _ = Cjk.shape
    lam = np.zeros((Ncfg, Lt, N))
    C0 = np.mean(Cjk[:, t0, :, :], axis=0)
    C0 = 0.5 * (C0 + C0.T)
    C0 += reg * np.eye(N)
    # Cholesky factorization
    evals = np.linalg.eigvalsh(C0)
    print("C0 eigenvals:", evals)
    try:
        L = la.cholesky(C0, lower=True)
    except la.LinAlgError:
        print("Cholesky failed — increasing regularization")
        C0 += 1e-8 * np.eye(N)
        L = la.cholesky(C0, lower=True)
    Linv = la.inv(L)
    for k in range(Ncfg):
        Cb = Cjk[k]
        for t in range(Lt):
            Ct = 0.5 * (Cb[t] + Cb[t].T)
            # standard eigenproblem
            M = Linv @ Ct @ Linv.T
            w, _ = la.eigh(M)
            # Sort descending (largest eig == ground state)
            lam[k, t] = np.sort(w)[::-1].real
    return lam


# def solve_gevp_jack(Cjk, t0):
#     Ncfg, Lt, N, _ = Cjk.shape
#     lam = np.zeros((Ncfg, Lt, N))
#     for k in range(Ncfg):
#         Cb = Cjk[k]
#         C0 = Cb[t0]
#         for t in range(Lt):
#             w, _ = la.eigh(Cb[t], C0)
#             idx = np.argsort(w)[::-1]
#             lam[k, t] = w[idx].real
#     return lam

# def solve_gevp_jack(Cjk, t0):
#     """USE MEAN ONLY """
#     Ncfg, Lt, N, _ = Cjk.shape
#     lam = np.zeros((Ncfg, Lt, N))
#     # Use mean C(t0) as metric
#     C0_mean = np.mean(Cjk[:, t0, :, :], axis=0)
#     # SYMMETRY
#     #C0_mean = 0.5 * (C0_mean + C0_mean.T)
#     for k in range(Ncfg):
#         Cb = Cjk[k]
#         for t in range(Lt):
#             w, _ = la.eigh(Cb[t], C0_mean)
#             idx = np.argsort(w)[::-1]
#             lam[k, t] = w[idx].real
#     return lam