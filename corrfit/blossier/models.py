import numpy as np
import lsqfit

class LinearPlusExp(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, p_keys, constant_only=False, exp_t0=False):
        super().__init__(datatag)

        # account for offset from gathering eigenvalues starting at t=3
        self.t = np.array(t)
        self.p_keys = p_keys
        self.particle = datatag[1]
        self.src_snk = datatag[2]
        self.constant_only = constant_only
        self.exp_t0 = exp_t0

    
    def fitfcn(self, p, t=None):
        # Fit function:
        #    \xi(t) = \xi ( 1 + \delta exp(-(E_{N+1} - E_n) t) )

        if t is None:
            t = self.t

        if self.exp_t0:
            t = np.round((t+0.5)/2)

        if self.constant_only:
            output = np.repeat(p[self.p_keys['xi']], len(t))
        else:
            # ensure the gap is positive
            # if p[self.p_keys['lambda_N']] < p[self.p_keys['lambda']]:
            #     p[self.p_keys['lambda']], p[self.p_keys['lambda_N']] = p[self.p_keys['lambda_N']],  p[self.p_keys['lambda']]
            #     if self.p_keys['xi'] == self.p_keys['lambda']:
            #         p[self.p_keys['xi']] = p[self.p_keys['lambda']]

            output = p[self.p_keys['xi']] *(1 +p[self.p_keys['delta']] *np.exp(-(p[self.p_keys['lambda_N']]- p[self.p_keys['lambda']]) *t))
        
        return output
    

    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior


    def builddata(self, data):
        if self.exp_t0:
            #return data[(self.particle, self.src_snk)][self.t - 3]
            return data[(self.particle, self.src_snk)][self.t - 4]
        else:
            return data[(self.particle, self.src_snk)][self.t - 4]