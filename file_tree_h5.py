import h5py
import numpy as np 
with h5py.File("spec_b3.4-s32t64_20260225-1105.h5", "r") as f:
    a = f["axial_T1g/charm_charm/t0avg/Matrix"][:]
    p = f["pseudoscalar_A1u/charm_charm/t0avg/Matrix"][:]
print(np.allclose(a, p))