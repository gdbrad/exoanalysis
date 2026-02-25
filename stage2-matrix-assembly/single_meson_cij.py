import numpy as np
import h5py
import glob
import os
import argparse
import os
from pathlib import Path

def discover_operator_basis(example_file, irrep_name):
    with h5py.File(example_file, "r") as f:
        pairs = list(f[irrep_name].keys())

    ops = set()
    for name in pairs:
        if "__X__" not in name:
            continue
        op1, op2 = name.split("__X__")
        ops.add(op1)
        ops.add(op2)

    return sorted(list(ops))

# def drop_bad_configs(Ccfg, cfg_numbers, bad_cfg):
#     if bad_cfg is None:
#         return Ccfg, cfg_numbers
#     bad = (cfg_numbers == bad_cfg)
#     print("dropping configs:", cfg_numbers[bad])
#     keep = ~bad
#     return Ccfg[keep], cfg_numbers[keep]

# Build C_ij matrix (single meson version)

def build_Cij(stage1_dir, irrep_name, output_file):

    cfg_files = sorted(glob.glob(os.path.join(stage1_dir, "cfg*.h5")))
    ncfg = len(cfg_files)
    if ncfg == 0:
        raise RuntimeError("No cfg files found.")
    operator_list = discover_operator_basis(cfg_files[0], irrep_name)
    nops = len(operator_list)
    op_index = {op: i for i, op in enumerate(operator_list)}

    with h5py.File(cfg_files[0], "r") as f:
        first_dataset = next(v for k, v in f[irrep_name].items()if "__X__" in k)
        Ntsrc, LT = first_dataset.shape
    print(f"Found {ncfg} configs")
    print(f"Operators: {nops}")
    print(f"Ntsrc: {Ntsrc}, Lt: {LT}")

    C = np.zeros((ncfg, Ntsrc, nops, nops, LT), dtype=np.complex128)

    for cfg_idx, fname in enumerate(cfg_files):
        print(f"[{cfg_idx+1}/{ncfg}] {fname}")
        with h5py.File(fname, "r") as f:
            for pair_name, dataset in f[irrep_name].items():
                if "__X__" not in pair_name:
                    continue
                op1, op2 = pair_name.split("__X__")
                i = op_index[op1]
                j = op_index[op2]
                data = dataset[:]  # (Ntsrc, Lt)
                C[cfg_idx, :, i, j, :] = data
                # reconstruct hermitian conj
                if i != j:
                    C[cfg_idx, :, j, i, :] = np.conjugate(data)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("Cij", data=C)
        f.create_dataset("operators",data=np.array(operator_list, dtype="S"))
    print("Stage 2 complete (single meson).")

def discover_blocks(example_file):
    """
    Return list of block names in a Stage-1 file.
    """
    with h5py.File(example_file, "r") as f:
        return list(f.keys())
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", required=True,
                        help="Directory containing cfgXXXX.h5 files")
    parser.add_argument("--outdir", required=True,
                        help="Output dir for Stage-2 files")
    parser.add_argument("--block", default=None,
                        help="Specific block (irrep_name) to process "
                             "if null then process all blocks")
    args = parser.parse_args()
    stage1_dir = Path(args.stage1_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_files = sorted(stage1_dir.glob("cfg*.h5"))
    if len(cfg_files) == 0:
        raise RuntimeError("No cfg*.h5 files found in Stage-1 directory.")
    example_file = str(cfg_files[0])
    if args.block is not None:
        blocks = [args.block]
    else:
        blocks = discover_blocks(example_file)
    print("Blocks to process:")
    for b in blocks:
        print("  ", b)
    for block_name in blocks:
        print("\n=====================================")
        print(f"Processing block: {block_name}")
        print("=====================================")
        output_file = outdir / f"Cij_{block_name}.h5"
        build_Cij(
            stage1_dir=str(stage1_dir),
            irrep_name=block_name,
            output_file=str(output_file)
        )
    print("\nStage 2 complete.")
    
if __name__ == "__main__":
    main()