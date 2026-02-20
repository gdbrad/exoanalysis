import numpy as np
import lsqfit

class ConstantModel(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, p_keys={}):
        super().__init__(datatag)
        self.t = np.array(t)
        self.particle = datatag[1]
        self.src_snk = datatag[2]
        self.p_keys = p_keys


    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        return np.repeat(p[self.p_keys['c']], len(t))


    def buildprior(self, prior, mopt=None):
        return prior


    def builddata(self, data):
        return data[(self.particle, self.src_snk)][self.t]