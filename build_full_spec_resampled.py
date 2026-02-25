import numpy as np
import h5py
import glob
import os
import argparse
import time
timestr = time.strftime("%Y%m%d-%H%M")

parser = argparse.ArgumentParser()
parser.add_argument("--data",type=str,required=True,help="mounted cluster dir containing Cij_*.h5 files")
parser.add_argument("--outdir",type=str,default=".",help="path to outupt spectrum file")
parser.add_argument("--ens",type=str,required=True,help="ens name (e.g. b3.4-s32t64)")
parser.add_argument("--bad_idx", type=int, default=None,help="config idx (number) to drop 3400 is 58")
parser.add_argument("--stat", choices=["bs", "jn"],default="jn")
parser.add_argument("--nboot",type=int,default=500)
args = parser.parse_args()

INPUT_DIR = args.data
OUTPUT_FILE = os.path.join(args.outdir,f"spec_{args.ens}_{timestr}.h5")
NBOOT = args.nboot
TSRC_STEP = 8
DROP_FIRST_TSRC = True

def drop_bad_configs(Ccfg, cfg_numbers, bad_cfg):
    if bad_cfg is None:
        return Ccfg, cfg_numbers
    bad = (cfg_numbers == bad_cfg)
    print("Dropping configs:", cfg_numbers[bad])
    keep = ~bad
    return Ccfg[keep], cfg_numbers[keep]

def drop_bad_index(Ccfg, bad_index):
    if bad_index is None:
        return Ccfg
    print("dropping config at index:", bad_index)
    tmp = np.ones(Ccfg.shape[0], dtype=bool)
    tmp[bad_index] = False
    return Ccfg[tmp]

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

def bs_resample(Ccfg, nboot):
    Ncfg = Ccfg.shape[0]
    rng = np.random.default_rng()
    Cboot = np.zeros((nboot,) + Ccfg.shape[1:])
    for b in range(nboot):
        idx = rng.integers(0, Ncfg, size=Ncfg)
        Cboot[b] = np.mean(Ccfg[idx], axis=0)
    return Cboot

def jack_resample(Ccfg):
    """
    single elim jackknife samples 
    Returns shape (Ncfg, Lt, Nops, Nops)
    """
    Ncfg = Ccfg.shape[0]
    Cjk = np.zeros_like(Ccfg)
    for i in range(Ncfg):
        Cjk[i] = np.mean(np.delete(Ccfg, i, axis=0), axis=0)
    return Cjk
####################### main ########################################

#input_files = sorted(glob.glob("Cij_*_*.h5"))
input_files = glob.glob(os.path.join(INPUT_DIR, "**/Cij_*_*.h5"),
          recursive=True)
#input_files = sorted(
 #   glob.glob(os.path.join(INPUT_DIR, "Cij_*_*.h5")))

print("found files:")
for f in input_files:
    print(" ", f)

with h5py.File(OUTPUT_FILE, "w") as fout:
    for fname in input_files:
        print("\nprocessing:", fname)
        # parse channel + flavor
        # format: Cij_<channel>_<flavor>.h5
        base = os.path.basename(fname).replace("Cij_", "").replace(".h5", "")
        parts = base.split("_")
        channel = "_".join(parts[:2])
        flavor  = "_".join(parts[2:])
        print(" channel:", channel)
        print(" flavor:", flavor)

        # load stage-2 file
        with h5py.File(fname, "r") as f:
            C = f["Cij"][:]  # (Ncfg, Ntsrc, Nops, Nops, Lt)
            ops = [o.decode() for o in f["operators"][:]]
        print("  raw shape:", C.shape)
        Ccfg = tsrc_average(C)
        print("  after tsrc avg:", Ccfg.shape)
        print("shape before drop:", Ccfg.shape)
        if args.bad_idx:
            print("mean norm of cfg 58:",np.mean(np.abs(Ccfg[58])))
            Ccfg = drop_bad_index(Ccfg, args.bad_idx)
            print("  after drop:", Ccfg.shape)
        # if "cfgs" in f:
        #     cfg_numbers = f["cfgs"][:]
        # else:
        #     cfg_numbers = np.arange(Ccfg.shape[0])
        # # drop bad config
        # Ccfg, cfg_numbers = drop_bad_configs(Ccfg, cfg_numbers, args.bad_cfg)
        # print("  shape after drop:", Ccfg.shape)

        ################### resmample #####################################
        if args.stat == "bs":
            Cstat = bs_resample(Ccfg, args.nboot)
            print("after bootstrap:", Cstat.shape)
        else:
            Cstat = jack_resample(Ccfg)
            print("after jackknife:", Cstat.shape)

        grp_channel = fout.require_group(channel)
        grp_flavor  = grp_channel.require_group(flavor)
        grp_t0      = grp_flavor.require_group("t0avg")
        grp_t0.create_dataset("Matrix", data=Cstat)
        grp_t0.create_dataset("operators",data=np.array(ops, dtype="S"))
        grp_t0.attrs["stat_type"] = args.stat

print("\nwrote out to:", OUTPUT_FILE)
