import numpy as np
import corrfit.base
import corrfit.constant.models

class Fitter(corrfit.base.Fitter):

    def __init__(self, data, prior, fit_args, p0=None, build_prior=True, svdcut=None):
        super().__init__(data, prior, fit_args, p0, build_prior, svdcut=svdcut)


    @property
    def models(self):
        models = np.array([])

        for part, src_snk in self.data:
            if self.fit_args[part]['perform_fit']:
                t_start = self.fit_args[part]['t_start']
                t_end = self.fit_args[part]['t_end']

                datatag = ('const', part, src_snk)

                models = np.append(models,
                    corrfit.constant.models.ConstantModel(datatag, t=range(int(t_start), int(t_end)), p_keys=self.p_keys[(part, src_snk)]))
                
        return models
    

    def _build_p_keys(self):
        # c::R1_0::s -> {('R1_0', 's'): {'c': 'c::R1_0::s'}}
        output = {}
        for key in self._input_prior:
            pkey, part, ss = key.split('::')
            if (part, ss) not in output:
                output[part, ss] = {}
            output[part, ss][pkey] = key

        return output

    
    def _build_prior(self, input_prior=None):
        if input_prior is None:
            input_prior = self._input_prior

        output = {}
        for key in input_prior:
            if key.startswith('c::'):
                output[key] = input_prior[key]
        return output
    