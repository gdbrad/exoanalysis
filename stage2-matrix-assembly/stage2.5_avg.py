import numpy as np
import h5py


BAD_CFG = 3400   # update if needed


def average_tsrc_matrix(C, tsrc_step=8, drop_first=True):
    """
    C shape: (Ncfg, Ntsrc, N, N, Nt)
    Returns: (Ncfg, N, N, Nt)
    """

    if drop_first:
        C = C[:, 1:, ...]

    Ncfg, Ntsrc, N, _, Nt = C.shape

    shifts = np.arange(Ntsrc) * tsrc_step
    rolled = np.empty_like(C)

    for k in range(Ntsrc):
        rolled[:, k] = np.roll(C[:, k], -shifts[k], axis=-1)

    return rolled.mean(axis=1)


def average_tsrc_single(C, tsrc_step=8, drop_first=True):
    """
    C shape: (Ncfg, Ntsrc, Nt)
    Returns: (Ncfg, Nt)
    """

    if drop_first:
        C = C[:, 1:, ...]

    Ncfg, Ntsrc, Nt = C.shape

    shifts = np.arange(Ntsrc) * tsrc_step
    rolled = np.empty_like(C)

    for k in range(Ntsrc):
        rolled[:, k] = np.roll(C[:, k], -shifts[k], axis=-1)

    return rolled.mean(axis=1)


def drop_bad_configs(C15, C6, CD, Cpi, cfg_numbers):

    bad_mask = (cfg_numbers == BAD_CFG)

    print("Dropping configs:", cfg_numbers[bad_mask])

    keep = ~bad_mask

    return (
        C15[keep],
        C6[keep],
        CD[keep],
        Cpi[keep],
        cfg_numbers[keep],
    )


def main():

    input_file  = "b3.4-s32t64.h5"
    output_file = "b3.4-stage3-input.h5"

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
