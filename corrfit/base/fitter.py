import numpy as np
import lsqfit
import lsqfitics
import copy
import gvar as gv
import abc
import warnings

class Fitter(abc.ABC):

    def __init__(self, data, prior, fit_args, p0=None, build_prior=True, svdcut=None):

        self.data = data
        self._input_prior = prior
        self.fit_args = fit_args
        
        self._input_p0 = p0
        self.build_prior = build_prior
        self._p_keys = None
        self._fit = None

        if svdcut is not None:
            self.svdcut = svdcut
        elif all([fit_args[p_ss]['svdcut'] is None for p_ss in fit_args if fit_args[p_ss]['perform_fit']]):
            self.svdcut = None
        else:
            self.svdcut = np.max([fit_args[p_ss]['svdcut'] for p_ss in fit_args 
                if fit_args[p_ss]['perform_fit'] and fit_args[p_ss]['svdcut'] is not None])


    def __str__(self):
        output = '~'*15 +' FIT SETTINGS ' + '~'*15 +'\n'
        output += str(self.fit_args)
        output += '\n'
        output += '~'*15 +' FIT RESULTS ' + '~'*15 +'\n'
        output += str(self.fit)
        return output


    @property
    @abc.abstractmethod
    def models(self):
        # override this class
        pass


    @abc.abstractmethod
    def _build_prior(self, **kwargs):
        pass

    # calculate gaussian approximate posterior/AIC per eq 44 of arxiv/2008.01069
    # Note that preferred models have a smaller GAP/AIC
    @property
    def BAIC(self):
        return self._fit.BAIC
    

    @property
    def dcut(self):
        dcut = np.sum(np.array([
            (len(self.data[p_ss])
            - (self.fit_args.get(p_ss)['t_end'] - self.fit_args.get(p_ss)['t_start'])
            )
            for p_ss in self.data
            if self.fit_args.get(p_ss)['perform_fit']]))
        return dcut


    @property
    def fit(self):
        if self._fit is None:
            if all([self.fit_args[p_ss]['perform_fit'] == False for p_ss in self.fit_args]):
                return None

            else:
                if any([self.fit_args[p_ss]['uncorrelated'] == True for p_ss in self.fit_args]):
                    warnings.warn("Warning: performing uncorrelated fit")
                    decorrelate = lambda dg : {k : gv.gvar(gv.mean(dg[k]), gv.sdev(dg[k])) for k in dg}
                    data = decorrelate(self.data)
                else:
                    data = self.data
                #
                fitter = lsqfit.MultiFitter(models=self.models)
                fit = fitter.lsqfit(data=data, prior=self.prior, p0=self.p0, 
                    svdcut=self.svdcut, noise=(True, False))
                self._fit = lsqfitics.from_fit(fit, dcut=self.dcut)
            
        return self._fit

    
    @property
    def p_keys(self):
        if self._p_keys is None:
            self._p_keys = self._build_p_keys()
        return copy.deepcopy(self._p_keys)


    @property
    def p0(self):
        if self._input_p0 is None:
            return None

        #return gv.mean(self._input_p0)
        return gv.mean(self._build_prior(self._input_p0))


    @property
    def posterior(self):
        output = {}
        temp_p = self.fit.p
        for part, _ in self.data:
            output[part] = {'_'.join(np.delete(p.split('::'), 1))  : temp_p[p] for p in temp_p if p.split('::')[1] == part}

        return output


    @property
    def prior(self):
        if self._input_prior is None:
            return None
        elif self.build_prior:
            return self._build_prior(self._input_prior)
        else:
            return copy.deepcopy(self._input_prior)
        
        
    def _build_p_keys(self):
        # e.g., 
        # 1) c::R1_0::s -> {('R1_0', 's'): {'c': 'c::R1_0'}}
        # 2) d::R1_0::(p,s) -> {('R1_0', ('p', 's'): {'d': 'd::R1_0::(p,s)'}}
        tuple_to_str =  lambda t : t if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'
        contains_str_or_int = lambda e, l : bool(e in [str(li) for li in l])

        output = {}
        for label in sorted(self._input_prior):
            
            param = label.split('::')[0]
            
            if len(label.split('::')) == 3:
                _, part, src_snk_smr = label.split('::')
                # src & snk specified
                if src_snk_smr in [tuple_to_str(s) for _, s in self.data]:
                    src_snk = next(s for _, s in self.data if tuple_to_str(s) == src_snk_smr)
                    if (part, src_snk) not in output:
                        output[(part, src_snk)] = {}
                    if param.startswith('log('):
                        param = param + ')'
                    output[(part, src_snk)][param] = label
                
                # only smr specified
                elif src_snk_smr in [si for _, s in self.data for si in s]:
                    smr = src_snk_smr
                    
                    for src_snk in [s for p, s in self.data if p == part and contains_str_or_int(smr, s)]:
                        if (part, src_snk) not in output:
                            output[(part, src_snk)] = {}
                        if str(src_snk[0]) == smr:
                            if param.startswith('log('):
                                output[(part, src_snk)][param+'_src'+')'] = label
                            else:
                                output[(part, src_snk)][param+'_src'] = label
                        if str(src_snk[1]) == smr:
                            if param.startswith('log('):
                                output[(part, src_snk)][param+'_snk'+')'] = label
                            else:
                                output[(part, src_snk)][param+'_snk'] = label

            elif len(label.split('::')) == 2:
                _, part = label.split('::')
                for src_snk in [s for p, s in self.data if p == part]:
                    if (part, src_snk) not in output:
                        output[(part, src_snk)] = {}
                    if param.startswith('log('):
                        param = param + ')'
                    output[(part, src_snk)][param] = label

        return output