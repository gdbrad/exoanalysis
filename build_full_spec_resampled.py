# build_full_spec_resampled.py
import numpy as np
import h5py
import glob
import os

# ==========================================================
# SETTINGS
# ==========================================================

import argparse

# ==========================================================
# ARGUMENT PARSER
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Mounted cluster directory containing Cij_*.h5 files"
)

parser.add_argument(
    "--outdir",
    type=str,
    default=".",
    help="Where to write the spectrum file"
)

parser.add_argument(
    "--ens",
    type=str,
    required=True,
    help="Ensemble name (e.g. b3.4-s32t64)"
)

parser.add_argument(
    "--nboot",
    type=int,
    default=500
)

args = parser.parse_args()

INPUT_DIR = args.data
OUTPUT_FILE = os.path.join(args.outdir,
                           f"spec_{args.ens}.h5")

NBOOT = args.nboot
TSRC_STEP = 8
DROP_FIRST_TSRC = True

# ENSEMBLE_NAME = "b3.4-s32t64"
# OUTPUT_FILE = f"spec_{ENSEMBLE_NAME}.h5"

# NBOOT = 500
# TSRC_STEP = 8
# DROP_FIRST_TSRC = True


# ==========================================================
# TSRC AVERAGING FUNCTION
# ==========================================================

def tsrc_average(C):
    """
    C shape:
        (Ncfg, Ntsrc, Nops, Nops, Lt)

    Returns:
        (Ncfg, Lt, Nops, Nops)
    """

    Ncfg, Ntsrc, Nops, _, Lt = C.shape
    Ccfg = np.zeros((Ncfg, Lt, Nops, Nops))

    for cfg in range(Ncfg):
        for i in range(Nops):
            for j in range(Nops):

                dataset = C[cfg, :, i, j, :]

                if DROP_FIRST_TSRC:
                    dataset = dataset[1:]

                ntsrc_eff = dataset.shape[0]
                avg = np.zeros(Lt, dtype=dataset.dtype)

                for k in range(ntsrc_eff):
                    shift = -(k + 1) * TSRC_STEP
                    avg += np.roll(dataset[k], shift)

                avg /= ntsrc_eff

                if np.max(np.abs(avg.imag)) < 1e-12:
                    avg = avg.real

                Ccfg[cfg, :, i, j] = avg

    return Ccfg


# ==========================================================
# BOOTSTRAP FUNCTION
# ==========================================================

def bootstrap_resample(Ccfg, nboot):

    Ncfg = Ccfg.shape[0]
    rng = np.random.default_rng()

    Cboot = np.zeros((nboot,) + Ccfg.shape[1:])

    for b in range(nboot):
        idx = rng.integers(0, Ncfg, size=Ncfg)
        Cboot[b] = np.mean(Ccfg[idx], axis=0)

    return Cboot


# ==========================================================
# MAIN
# ==========================================================



#input_files = sorted(glob.glob("Cij_*_*.h5"))
input_files = glob.glob(os.path.join(INPUT_DIR, "**/Cij_*_*.h5"),
          recursive=True)
#input_files = sorted(
 #   glob.glob(os.path.join(INPUT_DIR, "Cij_*_*.h5")))

print("Found files:")
for f in input_files:
    print(" ", f)

with h5py.File(OUTPUT_FILE, "w") as fout:

    for fname in input_files:

        print("\nProcessing:", fname)

        # parse channel + flavor
        # format: Cij_<channel>_<flavor>.h5
        base = os.path.basename(fname).replace("Cij_", "").replace(".h5", "")
        parts = base.split("_")

        channel = "_".join(parts[:2])
        flavor  = "_".join(parts[2:])

        print("  Channel:", channel)
        print("  Flavor:", flavor)

        # --------------------------------------------
        # load stage-2 file
        # --------------------------------------------

        with h5py.File(fname, "r") as f:
            C = f["Cij"][:]  # (Ncfg, Ntsrc, Nops, Nops, Lt)
            ops = [o.decode() for o in f["operators"][:]]

        print("  Raw shape:", C.shape)

        # --------------------------------------------
        # tsrc average
        # --------------------------------------------

        Ccfg = tsrc_average(C)

        print("  After tsrc avg:", Ccfg.shape)

        # --------------------------------------------
        # bootstrap
        # --------------------------------------------

        Cboot = bootstrap_resample(Ccfg, NBOOT)

        print("  After bootstrap:", Cboot.shape)

        # --------------------------------------------
        # write to output
        # --------------------------------------------

        grp_channel = fout.require_group(channel)
        grp_flavor  = grp_channel.require_group(flavor)
        grp_t0      = grp_flavor.require_group("t0avg")

        grp_t0.create_dataset("Matrix", data=Cboot)
        grp_t0.create_dataset("operators",
                              data=np.array(ops, dtype="S"))

print("\nFinished writing:", OUTPUT_FILE)
