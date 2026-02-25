import numpy as np
import h5py
import gvar as gv
import lsqfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

import gevp_spec

parser = argparse.ArgumentParser()
parser.add_argument("--specfile", required=True)
parser.add_argument("--t0", type=int, default=4)
parser.add_argument("--tmin", type=int, default=4)
parser.add_argument("--tmax", type=int, default=18)
args = parser.parse_args()

def print_tree(name, obj):
    print(name)

with h5py.File(args.specfile, "r") as f:
    f.visititems(print_tree)

    channels = list(f.keys())
    for channel in channels:
        for flavor in f[channel]:
            print("\n====================================")
            print("Channel:", channel)
            print("Flavor content :", flavor)
            # print("Matrix id:", f[channel][flavor]["t0avg"]["Matrix"])
            # print("Matrix id:", f[channel][flavor]["t0avg"]["Matrix"])
            dset = f[channel][flavor]["t0avg"]["Matrix"]
            print("PATH:", dset.name)
            Cjk = f[channel][flavor]["t0avg"]["Matrix"][:]
            Cjk = Cjk.real
            print("Object id:", id(Cjk))
            print("Sum:", np.sum(Cjk))
            print("Norm:", np.linalg.norm(Cjk))
            ops   = [o.decode() for o in f[channel][flavor]["t0avg"]["operators"][:]]
            print("Flavor:", flavor)
            print("Mean C[t=0]:", np.mean(Cjk[:,0,:,:]))
            print("Mean C[t=5]:", np.mean(Cjk[:,5,:,:]))
            print("First element:", Cjk[0,0,0,0])
            print("Last element:", Cjk[-1,-1,-1,-1])
            
            
            # RESTRICTING BASIS FOR NOW  (drop B and D)
            # TODO ADD BASIS RESTRICTION FLAG 
            keep = [i for i, op in enumerate(ops) if "nabla" in op or "none" in op]

            Cjk = Cjk[:, :, keep, :][:, :, :, keep]
            ops = [ops[i] for i in keep]
            Ncfg, Lt, Nops, _ = Cjk.shape
            Nboot, Lt, Nops, _ = Cjk.shape

            print("jn shape:", Cjk.shape)
            print("operators:", ops)
            # Cjk is jack samples: (Ncfg, Lt, Nops, Nops)
            data_dict = {}
            for i in range(Nops):
                for j in range(Nops):
                    key = f"{i}_{j}"
                    data_dict[key] = Cjk[:, :, i, j]
            Cavg = gv.dataset.avg_data(data_dict, bstrap=False)

            # rebuild matrix
            Cmat = np.zeros((Lt, Nops, Nops), dtype=object)
            for i in range(Nops):
                for j in range(Nops):
                    Cmat[:, i, j] = Cavg[f"{i}_{j}"]

            # # convert full matrix to gvar
            # Cmat = np.zeros((Lt, Nops, Nops), dtype=object)
            # for i in range(Nops):
            #     for j in range(Nops):
            #         Cmat[:, i, j] = bootstrap_to_gvar(Cjk[:, :, i, j])
            # plot diag correlators
            plt.figure()
            for i in range(Nops):
                C = Cmat[:, i, i]
                plt.errorbar(range(Lt),gv.mean(C),yerr=gv.sdev(C),fmt='o',label=ops[i])
            plt.yscale("log")
            plt.title(f"{channel} {flavor} diag")
            plt.legend()
            plt.savefig(f"plots/{channel}_{flavor}_diag.png")
            plt.close()
            # solve GEVP for bs 
            # lam_boot = solve_gevp_bootstrap(Cjk,args.t0)
            # lam0_boot = lam_boot[:,:,0]
            # lam0 = bootstrap_to_gvar(lam0_boot)

            lam_jk = gevp_spec.solve_gevp_jack(Cjk, args.t0)
            lam0_jk = lam_jk[:, :, 0]   # shape (Ncfg, Lt)
            # lam_dict = {"lam0": lam0_jk}
            # lam_avg = gv.dataset.avg_data(lam_dict, bstrap=False)
            # lam0 = lam_avg["lam0"]

            
            # plot principal correlators (jack)

            # convert ALL principal corrs to gvar
            lam_dict = {f"state{n}": lam_jk[:, :, n] for n in range(Nops)}
            lam_avg = gv.dataset.avg_data(lam_dict, bstrap=False)
            lam0 = lam_avg["state0"]
            plt.figure()
            for n in range(Nops):
                lam_gv = lam_avg[f"state{n}"]
                plt.errorbar(range(Lt),gv.mean(lam_gv),yerr=gv.sdev(lam_gv),fmt='o',label=f"state {n}")
            plt.yscale("log")
            plt.title(f"{channel} {flavor} principal")
            plt.legend()
            plt.savefig(f"plots/{channel}_{flavor}_principal.png")
            plt.close()

            # # plot principal correlators
            # old bs version below 
            # plt.figure()
            # for n in range(Nops):
            #     plt.errorbar(range(Lt),
            #                 np.mean(lam_boot[:, :, n], axis=0),
            #                 yerr=np.std(lam_boot[:, :, n], axis=0),
            #                 fmt='o',
            #                 label=f"state {n}")

            # plt.yscale("log")
            # plt.title(f"{channel} {flavor} principal")
            # plt.legend()
            # plt.savefig(f"plots/{channel}_{flavor}_principal.png")
            # plt.close()
            #########################################
            # effective masses (ground state)

            m_eff = gevp_spec.effective_mass(lam0)

            plt.figure()
            plt.errorbar(range(len(m_eff)),
                        gv.mean(m_eff),
                        yerr=gv.sdev(m_eff),
                        fmt='o')
            plt.title(f"{channel} {flavor} m_eff")
            plt.savefig(f"plots/{channel}_{flavor}_meff.png")
            plt.close()

            # Fit ground state
            def scan_fit_windows(lam0, tmin_list, tmax, prior, fcn):
                results = []
                for tmin in tmin_list:
                    t = np.arange(tmin, tmax)
                    try:
                        fit = lsqfit.nonlinear_fit(
                            data=(t, lam0[tmin:tmax]),
                            prior=prior,
                            fcn=fcn,
                            svdcut=1e-2
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
            
            # fit ground state
            def one_exp(t, p):
                return p["A"] * gv.exp(-p["E"] * t)

            prior = gv.BufferDict()
            prior["A"] = gv.gvar(1.0, 2.0)
            prior["E"] = gv.gvar(0.35, 0.3)
            t = np.arange(args.tmin, args.tmax)

            fit = lsqfit.nonlinear_fit(
                data=(t, lam0[args.tmin:args.tmax]),
                prior=prior,
                fcn=one_exp,
                svdcut=1e-2
            )
            fit_file = open('fits.txt','w')
            fit_file.write(str(fit.format(maxline=True)))
            fit_file.write(str(fit.p["E"]))
            print(fit)
            print("g.s. E =", fit.p["E"])
            fit_file.close()

            plt.figure()
            # data
            plt.errorbar(range(Lt),gv.mean(lam0),yerr=gv.sdev(lam0),fmt='o',label="principal")

            # fit curve
            t_fit = np.arange(args.tmin, args.tmax)
            y_fit = one_exp(t_fit, fit.p)
            plt.plot(t_fit, gv.mean(y_fit), label="fit")
            plt.fill_between(t_fit,gv.mean(y_fit) - gv.sdev(y_fit),gv.mean(y_fit) + gv.sdev(y_fit),alpha=0.3)
            plt.yscale("log")
            plt.legend()
            plt.title(f"{channel} {flavor} principal+fit")
            plt.savefig(f"plots/{channel}_{flavor}_principal_fit.png")
            plt.close()

            # eff mass + fit band
            m_eff = gevp_spec.effective_mass(lam0)
            plt.figure()
            plt.errorbar(range(len(m_eff)),gv.mean(m_eff),yerr=gv.sdev(m_eff),fmt='o',label="m_eff")
            Efit = fit.p["E"]
            plt.axhline(gv.mean(Efit),linestyle="--",label="fit E")
            plt.fill_between([0, Lt],gv.mean(Efit) - gv.sdev(Efit),gv.mean(Efit) + gv.sdev(Efit),alpha=0.2)
            plt.legend()
            plt.title(f"{channel} {flavor} m_eff+fit")
            plt.savefig(f"plots/{channel}_{flavor}_meff_fit.png")
            plt.close()

            # tmin stability scan
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
                plt.savefig(f"plots/{channel}_{flavor}_stability.png")
                plt.close()

                # chi2 plot
                plt.figure()
                chi2 = [r["chi2dof"] for r in results]
                plt.plot(tmins, chi2, 'o-')
                plt.axhline(1.0, linestyle="--")
                plt.xlabel("tmin")
                plt.ylabel("chi2/dof")
                plt.title(f"{channel} {flavor} chi2_scan")
                plt.savefig(f"plots/{channel}_{flavor}_chi2scan.png")
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