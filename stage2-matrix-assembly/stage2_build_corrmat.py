import numpy as np
import h5py
import glob
import os


def reconstruct_hermitian(matrix_upper, nops):
    LT = next(iter(matrix_upper.values())).shape[0]
    full = np.zeros((nops, nops, LT), dtype=np.float64)

    for (i, j), data in matrix_upper.items():
        full[i, j] = data
        if i != j:
            full[j, i] = data

    return full


def discover_operator_basis(example_file, irrep):
    with h5py.File(example_file, "r") as f:
        pairs = list(f[irrep].keys())

    ops = set()
    for name in pairs:
        op1, op2 = name.split("__X__")
        ops.add(op1)
        ops.add(op2)

    ops = sorted(list(ops))
    return ops


def build_Cij(stage1_dir, irrep, output_file):

    cfg_files = sorted(glob.glob(os.path.join(stage1_dir, "cfg*.h5")))
    ncfg = len(cfg_files)

    if ncfg == 0:
        raise RuntimeError("No cfg files found.")

    operator_list = discover_operator_basis(cfg_files[0], irrep)
    nops = len(operator_list)
    op_index = {op: i for i, op in enumerate(operator_list)}

    with h5py.File(cfg_files[0], "r") as f:
        first_pair = next(iter(f[irrep].values()))
        Ntsrc = first_pair["dimeson_15"].shape[0]
        LT    = first_pair["dimeson_15"].shape[1]

    print(f"Found {ncfg} configs")
    print(f"Operators: {nops}")
    print(f"Ntsrc: {Ntsrc}, Lt: {LT}")

    # Now KEEP tsrc
    C15 = np.zeros((ncfg, Ntsrc, nops, nops, LT))
    C6  = np.zeros((ncfg, Ntsrc, nops, nops, LT))
    CD  = np.zeros((ncfg, Ntsrc, LT))
    Cpi = np.zeros((ncfg, Ntsrc, LT))

    for cfg_idx, fname in enumerate(cfg_files):

        print(f"[{cfg_idx+1}/{ncfg}] {fname}")

        with h5py.File(fname, "r") as f:

            upper15 = {}
            upper6  = {}

            for pair_name, grp in f[irrep].items():

                op1, op2 = pair_name.split("__X__")
                i = op_index[op1]
                j = op_index[op2]

                # NO averaging
                c15 = grp["dimeson_15"][:]   # (Ntsrc, LT)
                c6  = grp["dimeson_6"][:]

                upper15[(i, j)] = c15
                upper6[(i, j)]  = c6

            # Reconstruct for each tsrc
            for k in range(Ntsrc):

                mat15 = {key: val[k] for key, val in upper15.items()}
                mat6  = {key: val[k] for key, val in upper6.items()}

                C15[cfg_idx, k] = reconstruct_hermitian(mat15, nops)
                C6[cfg_idx, k]  = reconstruct_hermitian(mat6, nops)

            CD[cfg_idx]  = grp["meson1"][:]   # (Ntsrc, LT)
            Cpi[cfg_idx] = grp["meson2"][:]

    with h5py.File(output_file, "w") as f:
        f.create_dataset("C15", data=C15, compression="gzip")
        f.create_dataset("C6",  data=C6,  compression="gzip")
        f.create_dataset("D",   data=CD,  compression="gzip")
        f.create_dataset("pi",  data=Cpi, compression="gzip")
        f.create_dataset("operators", data=np.array(operator_list, dtype="S"))

    print("Stage 2 complete.")
