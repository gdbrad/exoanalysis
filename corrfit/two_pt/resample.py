import functools
import gvar as gv

import corrfit.base
import corrfit.two_pt.fitter
import corrfit.two_pt.fit_args

class Resampler(corrfit.base.FitResampler):

    def __init__(self, data, prior, fargs_unfmt, 
            seed=None, n_copies=None, jackknife=True, model_avg_args=None, rw_factors=None, preprocessed=False): 

        super().__init__(
            data=data, prior=prior, fargs_unfmt=fargs_unfmt, seed=seed, n_copies=n_copies, 
            jackknife=jackknife, model_avg_args=model_avg_args, rw_factors=rw_factors,
            preprocessed=preprocessed)


    @functools.cached_property
    def fit_args(self):
        return corrfit.two_pt.fit_args.FitArgs(fargs_unfmt=self._fargs_unfmt)


    def make_fitter(self, data=None, prior=None, fit_args=None, 
            p0=None, build_prior=True, average=False):
        if data is None:
            if self.jackknife:
                data = self.resampled_data._jackknife(n=0)
            else:
                data = self.resampled_data._bootstrap(n=0)
        if prior is None:
            prior = self.prior
        if fit_args is None:
            fit_args = self.fit_args

        fitter = corrfit.two_pt.fitter.Fitter(
            data=data, 
            prior=prior,
            fit_args=fit_args,
            p0=p0,
            build_prior=build_prior
        )

        return fitter
    

