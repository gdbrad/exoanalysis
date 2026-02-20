import os
import h5py
import numpy as np
import gvar as gv 
import lsqfit 

import stage3_io
import stage3_gevp


# call gevp and instantiate object with raw corrs 

# solve for the optimized correlators with above 

class MesonTwoPt(lsqfit.MultiFitterModel):
    pass 

class GEVPRatio(lsqfit.MultiFitterModel):
    pass 




class Fitter:
    def __init__(self,
                 data,
                 prior,
                 fit_args):
        self.data = data
        self.prior = prior 
        self.fit_args = fit_args
    






# fit the denominator of ratio to get posterior 




# run fit (correlator_denom, prior, fitargs)

