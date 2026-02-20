import numpy as np
import corrfit.base
import corrfit.blossier.models

class Fitter(corrfit.base.Fitter):

    def __init__(self, data, prior, fit_args, p0=None, build_prior=True, svdcut=None):
        super().__init__(data, prior, fit_args, p0, build_prior, svdcut=svdcut)


    @property
    def dcut(self):
        dcut = super().dcut
        for part, src_snk in self.data:
            if self.fit_args.get((part, src_snk))['perform_fit']:
                if self.fit_args.get((part, src_snk))['exp_t0']:
                    t_start = self.fit_args.get((part, src_snk))['t_start']
                    t_end = self.fit_args.get((part, src_snk))['t_end']

                    dcut += np.floor((t_end - t_start) / 2)

        return dcut


    @property
    def models(self):
        models = np.array([])

        for part, src_snk in self.data:
            if (self.fit_args.get((part, src_snk))['perform_fit'] 
                    and int(src_snk) < self.fit_args.get((part, src_snk))['n_states']):
                t_start = self.fit_args.get((part, src_snk))['t_start']
                t_end = self.fit_args.get((part, src_snk))['t_end']
                constant_only = self.fit_args.get((part, src_snk))['constant_only']
                exp_t0 = self.fit_args.get((part, src_snk))['exp_t0']

                t = np.arange(int(t_start), int(t_end))
                if exp_t0:
                    t = t[::2]
                
                # offset = 0
                # if t[0] % 2 == 1:
                #     offset = 1

                datatag = ('linexp', part, src_snk)

                models = np.append(models,
                    corrfit.blossier.models.LinearPlusExp(
                        datatag, t=t,
                        p_keys=self.p_keys[(part, src_snk)], 
                        constant_only=constant_only, exp_t0=exp_t0))
                
        return models
    

    def _build_p_keys(self):
        # c::R1_0::s -> {('R1_0', 's'): {'c': 'c::R1_0::s'}}
        output = {}
        for key in self._input_prior:
            pkey, part, ss = key.split('::')
            if ss != 'N' and self.fit_args.get((part, ss))['perform_fit'] and int(ss) < self.fit_args.get((part, ss))['n_states']:
                ss = int(ss)
                xi_key = self.fit_args.get((part, ss))['xi']
                delta_key = self.fit_args.get((part, ss))['delta']
                lambda_key = self.fit_args.get((part, ss))['lambda']

                if (part, ss) not in output:
                    output[part, ss] = {}
            
                if pkey == xi_key:
                    output[part, ss]['xi'] = key
                if pkey == delta_key:
                    output[part, ss]['delta'] = key
                if pkey == lambda_key:
                    output[part, ss]['lambda'] = key
                    output[part, ss]['lambda_N'] = pkey+'::'+part+'::'+'N'

        return output

    
    def _build_prior(self, input_prior=None):
        if input_prior is None:
            input_prior = self._input_prior

        p_keys = self.p_keys

        prior_labels = []
        for part in p_keys:
            for pk in p_keys[part]:
                prior_labels.append(p_keys[part][pk])

        output = {}
        for l in np.unique(prior_labels):
            output[l] = input_prior[l]

        return output
    