import h5py
# with h5py.File("b3.4-s32t64.h5") as f:
#     print(f["C15"].shape)
import h5py
import time

# t0 = time.time()
# with h5py.File("b3.4-s32t64.h5","r") as f:
#     C15 = f["C15"][:]
# print("Load time:", time.time()-t0)
import h5py
import numpy as np

with h5py.File("b3.4-s32t64.h5","r") as f:
    C15 = f["C15"][:]

print("Any NaNs:", np.isnan(C15).any())

# find first NaN location
idx = np.argwhere(np.isnan(C15))
print("First NaN index:", idx[0])
