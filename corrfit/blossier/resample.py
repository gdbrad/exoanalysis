import functools
import gvar as gv

import corrfit.base
import corrfit.blossier.fit_args

class Resampler(corrfit.base.FitResampler):

    def __init__(self, data, prior, fargs_unfmt, seed=None, n_copies=3000, 
        jackknife=True, model_avg_args=None):
        super().__init__(data, prior, fargs_unfmt, seed=seed, n_copies=n_copies, 
            jackknife=jackknife, preprocessed=True, model_avg_args=model_avg_args)


    @functools.cached_property
    def fit_args(self):
        return corrfit.blossier.fit_args.FitArgs(fargs_unfmt=self._fargs_unfmt)


    def make_fitter(self, data=None, prior=None, p0=None, fit_args=None, 
            build_prior=True, average=False):

        if fit_args is None:
            fit_args = self.fit_args
        if data is None:
            if self.jackknife:
                data = self.resampled_data._jackknife(n=0)
            else:
                data = self.resampled_data._bootstrap(n=0)
        if prior is None:
            prior = self.prior

        #data = {(part, src_snk) : data[(part, src_snk)] 
        #        for part, src_snk in data if part in self.fit_args.particles}

        return corrfit.blossier.fitter.Fitter(
            data=data, 
            prior=prior,  
            fit_args=fit_args,
            p0=p0
        )