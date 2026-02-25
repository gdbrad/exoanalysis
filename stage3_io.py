import os
import h5py
import numpy as np
import corrfit.io


class InputOutput(corrfit.io.InputOutput):

    def __init__(self, h5_name: str,
                 project_path=None,
                 tsrc_step=8,
                 drop_first_tsrc=False):

        super().__init__(project_path=project_path)

        self.h5_name = h5_name
        self.file_h5 = os.path.join(self.project_path, self.h5_name)

        self.tsrc_step = tsrc_step
        self.drop_first_tsrc = drop_first_tsrc


    # ======================================================
    # Stage 2 loader (matrix-level)
    # ======================================================

    def load_stage2_matrices(self):

        with h5py.File(self.file_h5, "r") as f:

            C15 = f["C15"][:]   # (Ncfg, Ntsrc, Nops15, Nops15, Lt)
            C6  = f["C6"][:]
            CD  = f["D"][:]
            Cpi = f["pi"][:]

        print("Loaded Stage 2 matrices:")
        print("Raw min/max:", np.min(C6), np.max(C6))
        print("Raw NaNs:", np.isnan(C15).any())

        print("  C15:", C15.shape)
        print("  C6 :", C6.shape)
        print("  D  :", CD.shape)
        print("  pi :", Cpi.shape)

        return C15, C6, CD, Cpi


    # ======================================================
    # tsrc averaging (matrix)
    # ======================================================

    def average_tsrc_matrix(self, C):

        if self.drop_first_tsrc:
            C = C[:, 1:, ...]

        Ncfg, Ntsrc, Nops, _, Lt = C.shape

        Cavg = np.zeros((Ncfg, Nops, Nops, Lt))

        for cfg in range(Ncfg):
            if cfg % 10 == 0:
                print(f"Averaging cfg {cfg}/{Ncfg}", flush=True)
            for k in range(Ntsrc):
                shift = -k * self.tsrc_step
                Cavg[cfg] += np.roll(C[cfg, k], shift, axis=1)


                Cavg /= Ntsrc

        print("After tsrc avg (matrix):", Cavg.shape)
        return Cavg


    # ======================================================
    # tsrc averaging (single)
    # ======================================================

    def average_tsrc_single(self, C):

        if self.drop_first_tsrc:
            C = C[:, 1:, :]

        Ncfg, Ntsrc, Lt = C.shape

        Cavg = np.zeros((Ncfg, Lt))

        for cfg in range(Ncfg):
            for k in range(Ntsrc):
                shift = -(k + 1) * self.tsrc_step
                Cavg[cfg] += np.roll(C[cfg, k], shift)

        Cavg /= Ntsrc

        print("After tsrc avg (single):", Cavg.shape)
        return Cavg


    # ======================================================
    # Full Stage 3-ready loader
    # ======================================================

    def load_data_stage3(self):

        C15, C6, CD, Cpi = self.load_stage2_matrices()

        # C15 = self.average_tsrc_matrix(C15)
        # C6  = self.average_tsrc_matrix(C6)
        # CD  = self.average_tsrc_single(CD)
        # Cpi = self.average_tsrc_single(Cpi)

        # bad = np.isnan(C15).any(axis=(1,2,3))
        # print("Dropping configs:", np.where(bad)[0])

        # C15 = C15[~bad]
        # C6  = C6[~bad]
        # CD  = CD[~bad]
        # Cpi = Cpi[~bad]


        return C15, C6, CD, Cpi
