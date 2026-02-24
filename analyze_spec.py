import numpy as np
import h5py
import gvar as gv
import lsqfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--specfile", required=True)
parser.add_argument("--t0", type=int, default=4)
parser.add_argument("--tmin", type=int, default=4)
parser.add_argument("--tmax", type=int, default=18)
args = parser.parse_args()

def bootstrap_to_gvar(Cboot):
    """
    Convert bootstrap samples to gvar array.
    Cboot shape: (Nboot, Lt)
    """
    mean = np.mean(Cboot, axis=0)
    cov  = np.cov(Cboot.reshape(Cboot.shape[0], -1),
                  rowvar=False)
    return gv.gvar(mean, cov)


def effective_mass(C):
    return gv.log(C[:-1] / C[1:])


# def solve_gevp(Cmat, t0):
#     """
#     Cmat: gvar array shape (Lt, N, N)
#     """
#     Lt, N, _ = Cmat.shape

#     evals = []

#     C0 = gv.mean(Cmat[t0])
#     C0inv = np.linalg.inv(C0)

#     for t in range(Lt):
#         M = C0inv @ gv.mean(Cmat[t])
#         w, v = np.linalg.eig(M)
#         idx = np.argsort(w)[::-1]
#         evals.append(w[idx])

#     return np.array(evals)  # (Lt, N)

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
        C0inv = np.linalg.inv(C0)

        for t in range(Lt):
            M = C0inv @ Cb[t]
            w, _ = np.linalg.eig(M)
            idx = np.argsort(w)[::-1]
            lam_boot[b, t] = w[idx].real

    return lam_boot

with h5py.File(args.specfile, "r") as f:
    channels = list(f.keys())
    for channel in channels:
        for flavor in f[channel]:

            print("\n====================================")
            print("Channel:", channel)
            print("Flavor :", flavor)

            Cboot = f[channel][flavor]["t0avg"]["Matrix"][:]
            Cboot = Cboot.real
            ops   = [o.decode() for o in
                     f[channel][flavor]["t0avg"]["operators"][:]]

            Nboot, Lt, Nops, _ = Cboot.shape

            print("Bootstrap shape:", Cboot.shape)
            print("Operators:", ops)

            # convert full matrix to gvar
            Cmat = np.zeros((Lt, Nops, Nops), dtype=object)

            for i in range(Nops):
                for j in range(Nops):
                    Cmat[:, i, j] = bootstrap_to_gvar(
                        Cboot[:, :, i, j]
                    )

            # plot diag correlators

            plt.figure()
            for i in range(Nops):
                C = Cmat[:, i, i]
                plt.errorbar(range(Lt),
                             gv.mean(C),
                             yerr=gv.sdev(C),
                             fmt='o',
                             label=ops[i])

            plt.yscale("log")
            plt.title(f"{channel} {flavor} diag")
            plt.legend()
            plt.savefig(f"{channel}_{flavor}_diag.png")
            plt.close()
            # solve GEVP
            lam_boot = solve_gevp_bootstrap(Cboot,args.t0)
            lam0_boot = lam_boot[:,:,0]
            lam0 = bootstrap_to_gvar(lam0_boot)

            # plot principal correlators
            plt.figure()
            for n in range(Nops):
                plt.errorbar(range(Lt),
                            np.mean(lam_boot[:, :, n], axis=0),
                            yerr=np.std(lam_boot[:, :, n], axis=0),
                            fmt='o',
                            label=f"state {n}")

            plt.yscale("log")
            plt.title(f"{channel} {flavor} principal")
            plt.legend()
            plt.savefig(f"{channel}_{flavor}_principal.png")
            plt.close()
            # --------------------------------------------
            # Effective masses (ground state)
            # --------------------------------------------

            m_eff = effective_mass(lam0)

            plt.figure()
            plt.errorbar(range(len(m_eff)),
                        gv.mean(m_eff),
                        yerr=gv.sdev(m_eff),
                        fmt='o')
            plt.title(f"{channel} {flavor} m_eff")
            plt.savefig(f"{channel}_{flavor}_meff.png")
            plt.close()
            # --------------------------------------------
            # Fit ground state
            # --------------------------------------------

            def scan_fit_windows(lam0, tmin_list, tmax, prior, fcn):
                results = []

                for tmin in tmin_list:
                    t = np.arange(tmin, tmax)

                    try:
                        fit = lsqfit.nonlinear_fit(
                            data=(t, lam0[tmin:tmax]),
                            prior=prior,
                            fcn=fcn,
                            svdcut=1e-4  # less aggressive
                        )

                        results.append({
                            "tmin": tmin,
                            "E": fit.p["E"],
                            "chi2dof": fit.chi2/fit.dof,
                            "Q": fit.Q
                        })

                    except Exception as e:
                        print("Fit failed for tmin =", tmin)

                return results
            
            # --------------------------------------------
            # Fit ground state
            # --------------------------------------------

            def one_exp(t, p):
                return p["A"] * gv.exp(-p["E"] * t)

            prior = gv.BufferDict()
            prior["A"] = gv.gvar(1.0, 2.0)
            prior["E"] = gv.gvar(0.35, 0.3)

            # ----------------------------
            # Single chosen fit
            # ----------------------------
            t = np.arange(args.tmin, args.tmax)

            fit = lsqfit.nonlinear_fit(
                data=(t, lam0[args.tmin:args.tmax]),
                prior=prior,
                fcn=one_exp,
                svdcut=1e-4
            )

            print(fit)
            print("Ground state E =", fit.p["E"])

            # ----------------------------
            # Plot correlator + fit band
            # ----------------------------

            plt.figure()

            # data
            plt.errorbar(
                range(Lt),
                gv.mean(lam0),
                yerr=gv.sdev(lam0),
                fmt='o',
                label="principal"
            )

            # fit curve
            t_fit = np.arange(args.tmin, args.tmax)
            y_fit = one_exp(t_fit, fit.p)

            plt.plot(t_fit, gv.mean(y_fit), label="fit")
            plt.fill_between(
                t_fit,
                gv.mean(y_fit) - gv.sdev(y_fit),
                gv.mean(y_fit) + gv.sdev(y_fit),
                alpha=0.3
            )

            plt.yscale("log")
            plt.legend()
            plt.title(f"{channel} {flavor} principal+fit")
            plt.savefig(f"{channel}_{flavor}_principal_fit.png")
            plt.close()

            # ----------------------------
            # Effective mass + fit band
            # ----------------------------

            m_eff = effective_mass(lam0)

            plt.figure()
            plt.errorbar(
                range(len(m_eff)),
                gv.mean(m_eff),
                yerr=gv.sdev(m_eff),
                fmt='o',
                label="m_eff"
            )

            Efit = fit.p["E"]

            plt.axhline(
                gv.mean(Efit),
                linestyle="--",
                label="fit E"
            )

            plt.fill_between(
                [0, Lt],
                gv.mean(Efit) - gv.sdev(Efit),
                gv.mean(Efit) + gv.sdev(Efit),
                alpha=0.2
            )

            plt.legend()
            plt.title(f"{channel} {flavor} m_eff+fit")
            plt.savefig(f"{channel}_{flavor}_meff_fit.png")
            plt.close()

            # ----------------------------
            # tmin stability scan
            # ----------------------------

            tmin_list = range(3, 10)
            results = scan_fit_windows(lam0, tmin_list, args.tmax, prior, one_exp)

            if len(results) > 0:
                plt.figure()

                Evals = [gv.mean(r["E"]) for r in results]
                Eerrs = [gv.sdev(r["E"]) for r in results]
                tmins = [r["tmin"] for r in results]

                plt.errorbar(tmins, Evals, yerr=Eerrs, fmt='o')
                plt.xlabel("tmin")
                plt.ylabel("E")
                plt.title(f"{channel} {flavor} stability")
                plt.savefig(f"{channel}_{flavor}_stability.png")
                plt.close()

                # chi2 plot
                plt.figure()
                chi2 = [r["chi2dof"] for r in results]
                plt.plot(tmins, chi2, 'o-')
                plt.axhline(1.0, linestyle="--")
                plt.xlabel("tmin")
                plt.ylabel("chi2/dof")
                plt.title(f"{channel} {flavor} chi2_scan")
                plt.savefig(f"{channel}_{flavor}_chi2scan.png")
                plt.close()
                        

            
            # t = np.arange(args.tmin, args.tmax)

            # prior = gv.BufferDict()
            # prior["A"]  = gv.gvar(1.0, 2.0)
            # #prior["B"]  = gv.gvar(0.5, 2.0)
            # prior["E"]  = gv.gvar(0.35, 0.3)
            # #prior["dE"] = gv.gvar(-0.5, 1.0)

            # def one_exp(t, p):
            #     return p["A"] * gv.exp(-p["E"] * t)

            # # def two_exp(t, p):
            # #     E1 = p["E"]
            # #     E2 = p["E"] + gv.exp(p["dE"])

            # #     return (p["A"] * gv.exp(-E1 * t) +
            # #             p["B"] * gv.exp(-E2 * t))
            # print(gv.mean(m_eff[5:12]))

            # fit = lsqfit.nonlinear_fit(
            #     data=(t, lam0[args.tmin:args.tmax]),
            #     prior=prior,
            #     fcn=one_exp,
            #     svdcut=1e-3
            # )

            # print(fit)
            # print("Ground state E =", fit.p["E"])