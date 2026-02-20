import numpy as np
import lsqfit

# This class is needed to instantiate an object for lsqfit.MultiFitter
# Used for particles that obey fermi-dirac statistics
class BaryonModel(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, n_states, overlap=None, use_log_dE=None, t0_offset=None, p_keys={}):
        super(BaryonModel, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.t0_offset = t0_offset
        self.n_states = n_states
        self.particle = datatag[1]
        self.src_snk = datatag[2]
        self.overlap = overlap
        self.use_log_dE = use_log_dE
        self.p_keys = p_keys


    def fitfcn(self, p, t=None):
        
        if t is None:
            t = self.t
        t = np.array(t)

        t0 = self.t0_offset
        if t0 is None:
            t0 = 0
        
        if self.overlap == 'ZZ':
            wf = p[self.p_keys['Z_src']] *p[self.p_keys['Z_snk']]
        else:
            wf = p[self.p_keys['wf']]

        #print(self.particle)
        E = []
        for n in range(self.n_states+1):
            if 'E'+str(n) in self.p_keys:
                E.append(p[self.p_keys['E'+str(n)]])

        if len(E) < self.n_states:
            if self.use_log_dE:
                En = np.sum(E)
                dE = np.exp(p[self.p_keys['log(dE)']])
                E = E + [En + dE[j] for j in range(self.n_states-len(E))]
            else:
                En = np.sum(E)
                dE = p[self.p_keys['dE']]
                E = E + [En + dE[j] for j in range(self.n_states-len(E))]

        wf, E = np.array(wf), np.array(E)
        return np.sum(wfi *np.exp(-Ei *(t-t0)) for wfi, Ei in zip(wf, E))
    

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior


    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[(self.particle, self.src_snk)][self.t]


    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t
        t = np.array(t)
        
        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))


    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        t = np.array(t)
        
        return np.exp(self.fcn_effective_mass(p, t)*t) * self.fitfcn(p, t)


# Used for particles that obey bose-einstein statistics
# Definitely doesn't work atm
class MesonModel(lsqfit.MultiFitterModel):
    ####
    # Warning: this model doesn't currently work!!
    ####
    def __init__(self, datatag, t, t_period, n_states, overlap=None, use_log_dE=None, p_keys={}):
        super(MesonModel, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.t_period = t_period
        self.n_states = n_states
        self.particle = datatag[1]
        self.src_snk = datatag[2]
        self.overlap = overlap
        self.use_log_dE = use_log_dE
        self.p_keys = p_keys


    def fitfcn(self, p, t=None):

        if t is None:
            t = self.t

        if self.overlap == 'ZZ':
            E0 = p['E0']
        else:
            E0 = np.exp(p['log(E0)'])

        if self.overlap == 'ZZ':
            zz = p['Z_'+str(self.src_snk[0])] *p['Z_'+(self.src_snk[1])]

            output = zz[0] *(np.exp( -E0 *t ) + np.exp( -E0 *(self.t_period - t) ))
            for j in range(1, self.n_states):
                dE = np.exp(p['log(dE)'])
                E_j = E0 + np.sum([dE[k] for k in range(j)], axis=0)
                output = output + zz[j] *(np.exp( -E_j *t ) + np.exp( -E_j * (self.t_period - t) ))
        else:
            # format tuple as string
            if isinstance(self.src_snk, str):
                wf = p['wf_'+self.src_snk]
            else:
                wf = p['wf_'+'(' + ','.join([str(s) for s in list(self.src_snk)]) + ')']

            output = wf[0] * np.cosh( E0 * (t - self.t_period/2.0) )
            for j in range(1, self.n_states):
                dE = np.exp(p['log(dE)'])
                E_j = E0 + np.sum([dE[k] for k in range(j)], axis=0)
                output = output + wf[j] *np.cosh( E_j *(t - self.t_period/2.0) )

        return output


    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[(self.particle, self.src_snk)][self.t]


    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t
        return np.arccosh((self.fitfcn(p, t-1) + self.fitfcn(p, t+1))/(2*self.fitfcn(p, t)))


    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        if self.overlap == 'ZZ':
            return self.fitfcn(p, t) / (np.exp(-self.fcn_effective_mass(p, t) *t) + np.exp(-self.fcn_effective_mass(p, t) *(self.t_period - t)))
        else:
            return 1 / np.cosh(self.fcn_effective_mass(p, t)*(t - self.t_period/2)) * self.fitfcn(p, t)