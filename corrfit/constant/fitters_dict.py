import corrfit.base
import corrfit.constant.fitter

class FittersDict(corrfit.base.FittersDict):

    def __init__(self, data, prior, svdcut=None):
        super().__init__(data, prior, svdcut=svdcut)


    def _make_fitter(self, fit_args):
        data = {}
        data = {(part, src_snk) : self.data[(part, src_snk)] 
                for part, src_snk in self.data if part in fit_args}

        return corrfit.constant.fitter.Fitter(
            data=data, 
            prior=self.prior,  
            fit_args=fit_args,
            svdcut=self.svdcut
        )
    

    def __getitem__(self, fit_args):
        if isinstance(fit_args, dict):
            fit_args = corrfit.constant.fit_args.FitArgs(fit_args)
        return super().__getitem__(fit_args)