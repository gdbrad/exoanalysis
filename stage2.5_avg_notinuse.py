import numpy as np
import h5py


BAD_CFG = 3400   # update if needed


def average_tsrc_matrix(C, tsrc_step=8):
    """
    C shape: (Ncfg, Ntsrc, N, N, Nt)
    """

    # Drop corrupted tsrc=0
    C = C[:, 1:, ...]   # remove first source

    Ncfg, Ntsrc_eff, N, _, Nt = C.shape

    avg = np.zeros((Ncfg, N, N, Nt), dtype=C.dtype)

    for k in range(Ntsrc_eff):
        shift = -(k + 1) * tsrc_step
        avg += np.roll(C[:, k, :, :, :], shift,axis=-1)

    avg /= Ntsrc_eff

    return avg

def average_tsrc_single(C, tsrc_step=8):
    """
    C shape: (Ncfg, Ntsrc, Nt)
    """

    C = C[:, 1:, ...]   # drop tsrc=0

    Ncfg, Ntsrc_eff, Nt = C.shape

    avg = np.zeros((Ncfg, Nt), dtype=C.dtype)

    for k in range(Ntsrc_eff):
        shift = -(k + 1) * tsrc_step
        avg += np.roll(C[:, k, :], shift,axis=-1)

    avg /= Ntsrc_eff

    return avg


BAD_INDEX = 58   # verified bad cfg3400 index

# def drop_bad_configs(C15, C6, CD, Cpi, cfg_numbers):

#     print("Dropping config index:", BAD_INDEX)

#     C15 = np.delete(C15, BAD_INDEX, axis=0)
#     C6  = np.delete(C6,  BAD_INDEX, axis=0)
#     CD  = np.delete(CD,  BAD_INDEX, axis=0)
#     Cpi = np.delete(Cpi, BAD_INDEX, axis=0)
#     cfg_numbers = np.delete(cfg_numbers, BAD_INDEX, axis=0)

#     return C15, C6, CD, Cpi, cfg_numbers

def drop_bad_configs(C15, C6, CD, Cpi, cfg_numbers):

    nan_cfgs = np.unique(np.argwhere(np.isnan(C15))[:,0])

    print("Dropping cfg indices:", nan_cfgs)

    keep = np.ones(C15.shape[0], dtype=bool)
    keep[nan_cfgs] = False

    return (
        C15[keep],
        C6[keep],
        CD[keep],
        Cpi[keep],
        cfg_numbers[keep],
    )


def main():

    input_file  = "stage2-matrix-assembly/b3.4-s32t64.h5"
    output_file = "b3.4-stage3-input-fix.h5"

    with h5py.File(input_file, "r") as f:

        C15 = f["C15"][:]    # (Ncfg, Ntsrc, N, N, Nt)
        C6  = f["C6"][:]
        CD  = f["D"][:]
        Cpi = f["pi"][:]

        # If you stored cfg numbers in stage2
        if "cfgs" in f:
            cfg_numbers = f["cfgs"][:]
        else:
            # If not stored, assume sorted order
            cfg_numbers = np.arange(C15.shape[0])

    print("Raw shape:", C15.shape)

    # Drop bad config
    C15, C6, CD, Cpi, cfg_numbers = drop_bad_configs(
        C15, C6, CD, Cpi, cfg_numbers
    )

    print("After drop:", C15.shape)

    # Average tsrc
    C15_avg = average_tsrc_matrix(C15)
    C6_avg  = average_tsrc_matrix(C6)
    CD_avg  = average_tsrc_single(CD)
    Cpi_avg = average_tsrc_single(Cpi)

    print("After tsrc avg:", C15_avg.shape)

    # Transpose for GEVP: (Ncfg, Nt, N, N)
    C15_avg = np.transpose(C15_avg, (0, 3, 1, 2))
    C6_avg  = np.transpose(C6_avg,  (0, 3, 1, 2))

    print("Final shape for GEVP:", C15_avg.shape)

    # Save clean stage3 input
    with h5py.File(output_file, "w") as f:
        f.create_dataset("C15", data=C15_avg, compression="gzip")
        f.create_dataset("C6",  data=C6_avg,  compression="gzip")
        f.create_dataset("D",   data=CD_avg,  compression="gzip")
        f.create_dataset("pi",  data=Cpi_avg, compression="gzip")
        f.create_dataset("cfgs", data=cfg_numbers)

    print("Stage 2.5 complete.")


if __name__ == "__main__":
    main()
