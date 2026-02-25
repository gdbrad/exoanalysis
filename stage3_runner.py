import numpy as np
from corrfit.io import to_gvar
import matplotlib.pyplot as plt
import gvar as gv
import numpy as np


from stage3_io import InputOutput
#from stage3_gevp import rebuild_matrix, solve_gevp, project_correlator
#from fits import fit_correlator


def matrix_to_dict(C, tag):
    """
    C shape: (Ncfg, Nt, Nops, Nops)
    Output: (Ncfg, Nt) per (i,j)
    """

    Ncfg, Nt, Nops, _ = C.shape

    out = {}

    for i in range(Nops):
        for j in range(Nops):
            out[(tag, (i, j))] = C[:, :, i, j]

    return out


class Stage3Runner:

    def __init__(self, project_path, h5_name,ensemble):

        self.io = InputOutput(
            project_path=project_path,
            h5_name=h5_name,
            tsrc_step=8,
            drop_first_tsrc=True
        )

        self.ensemble = ensemble


    def run(self):

        # 1️⃣ Load tsrc-averaged data
        C15, C6, CD, Cpi = self.io.load_data_stage3()
        print(np.mean(C6[:, 0, 0, 0]))
        print(np.mean(C6[:, 0, 0, :10]))


        # 2️⃣ Convert to corrfit dict
        data_dict = {}
        #data_dict.update(matrix_to_dict(C15, "dpi_15"))
        #data_dict.update(matrix_to_dict(C6,  "dpi_6"))
        data_dict[("D", "local")] = CD
        data_dict[("pi", "local")] = Cpi

        # 3️⃣ Convert to gvar
        gdata = to_gvar(data_dict, decorrelate_keys=False)
        print(gdata)
        
        c = gdata[('pi','local')]
        means = gv.mean(c)
        #data = ptot["meson1_correlator"]
        plt.plot(np.arange(64), -means, '.')
        plt.yscale('log')
        plt.savefig('corr-pi.png')
        # results = {}

        # # 4️⃣ GEVP irreps
        # for irrep in ["dpi_15", "dpi_6"]:

        #     print(f"\nRunning GEVP for {irrep}")

        #     gevp_args = self.io.get_gevp_args(irrep, self.ensemble)
        #     t0   = gevp_args.get("t0", 3)
        #     tref = gevp_args.get("t_ref", 6)

        #     C = rebuild_matrix(gdata, irrep)
        #     evals, evecs = solve_gevp(C, t0)

        #     prior = self.io.get_prior(irrep, self.ensemble)
        #     fit_args = self.io.get_fit_args(irrep, self.ensemble)[irrep]

        #     irrep_results = {}

        #     for n in range(evecs.shape[2]):

        #         vec = evecs[tref, :, n]
        #         Cproj = project_correlator(C, vec)

        #         fit = fit_correlator(Cproj, prior, fit_args)

        #         print(f"{irrep} state {n} E =", fit.p["E"])

        #         irrep_results[f"state_{n}"] = fit.p

        #     results[irrep] = irrep_results

        # # 5️⃣ Single mesons
        # for meson in ["D", "pi"]:

        #     prior = self.io.get_prior(meson, self.ensemble)
        #     fit_args = self.io.get_fit_args(meson, self.ensemble)[meson]

        #     C = gdata[(meson, "local")]
        #     fit = fit_correlator(C, prior, fit_args)

        #     print(f"{meson} E =", fit.p["E"])

        #     results[meson] = fit.p

        # self.io.pickle_gvar_dict(results)

        # return results
if __name__ == "__main__":
    runner = Stage3Runner(
        project_path="/p/scratch/exflash/exotraction/",
        h5_name="b3.4-stage3-input-fix.h5",
        ensemble="b3.4"
    )
    runner.run()
