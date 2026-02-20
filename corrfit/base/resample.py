import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import hashlib
import tqdm
import lsqfit
import warnings
import multiprocess
import scipy.stats
import abc
import functools
import lsqfitics
import fnmatch

import corrfit.io

import warnings
warnings.simplefilter("ignore", UserWarning)

class RNG(np.random.Generator):
    def __init__(self, seed=None):
        self._seed_int = self._hash_seed(seed=seed)
        super().__init__(np.random.PCG64(np.random.SeedSequence(self._seed_int)))


    def reset(self, seed=None):
        if seed is not None:
            self._seed_int = self._hash_seed(seed=seed)
        super().__init__(np.random.PCG64(np.random.SeedSequence(self._seed_int)))

    
    def _hash_seed(self, seed=None, verbose=False):
        if seed is None:
            seed = str(np.random.randint(1e6))

        """Generate a random number generator based on a seed string."""
        # Over python iteration the traditional hash was changed. So, here we fix it to md5
        hash = hashlib.md5(seed.encode("utf-8")).hexdigest()  # Convert string to a hash
        seed_int = int(hash, 16) % (10 ** 6)  # Convert hash to an fixed size integer
        if verbose:
            print("Seed to md5 hash:", seed, "->", hash, "->", seed_int)

        return seed_int

class BootstrappedData(object):
    def __init__(self, data=None, rw_factors=None, preprocessed=False,
            seed=None, n_copies=None, n_configs=None, 
            decorrelate_keys=True):
        # preprocessed: has the data already been bootstrapped?

        if seed is None:
            seed = str(np.random.randint(1e6))

        if preprocessed and rw_factors is not None:
            raise ValueError('rw factors should be None!')

        if n_configs is None:
            n_configs = data[list(data)[0]].shape[0]
            if preprocessed:
                n_configs = n_configs - 1 # assume bs0 included

        if n_copies is None or preprocessed:
            n_copies = n_configs

        if preprocessed:
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=np.ComplexWarning)

                data_temp = {}
                for p_ss in data:
                    central = data[p_ss][0]
                    mean_rs = np.mean(data[p_ss][1:], axis=0)
                    data_temp[p_ss] = data[p_ss][1:] - mean_rs + central
                
                if decorrelate_keys:
                    covariance_frozen = gv.evalcov(
                        {p_ss : gv.dataset.avg_data(data_temp[p_ss], spread=True) for p_ss in data}) 
                else:
                    covariance_frozen = gv.evalcov(
                        gv.dataset.avg_data({p_ss : data_temp[p_ss] for p_ss in data}, spread=True)) 
                
        else:
            covariance_frozen = None

        self.covariance_frozen = covariance_frozen
        self.preprocessed = preprocessed
        self.n_configs = n_configs
        self.n_copies = n_copies
        self.data = data
        self.rw_factors = rw_factors
        self.seed = seed
        self.rng = RNG(seed=seed)
        self._bs_list = None


    def resample(self, means_only=False, as_gvar=False):
        self.rng.reset()

        for n in tqdm.tqdm(range(self.n_copies+1), desc='Bootstrapping: '):
            yield self._bootstrap(n=n, means_only=means_only, as_gvar=as_gvar)


    def save_resamples(self):
        output = {}
        for p_ss in self.data:
            shape = [self.n_copies+ 1] + list(self.data[p_ss].shape[1:])
            output[p_ss] = np.empty(shape)

        for j, data_rs in enumerate(self.resample(means_only=True)):
            for p_ss in data_rs:
                output[p_ss][j] = data_rs[p_ss]

        return output
    
    
    @property
    def bs_list(self):
        if self._bs_list is None:
            self._bs_list = self._make_bs_list(
                n_configs=self.n_configs, 
                n_copies=self.n_copies)
        return self._bs_list
    
    # taken from https://github.com/callat-qcd/bs_utils
    def _make_bs_list(self, n_configs, n_copies=None, seed=None, verbose=False):
        if n_copies is None:
            n_copies = n_configs

        # make BS list: [low, high)
        return self.rng.integers(low=0, high=n_configs, size=[n_copies, n_configs])


    def _bootstrap(self, n, means_only=False, as_gvar=False):
        '''
        Note that means_only==True iff as_gvar==False
        '''
        if self.preprocessed:
            # output = {p_ss: gv.gvar(self.data[p_ss][n, :], self.covariance_frozen[p_ss]) 
            #     for p_ss in self.data}
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=np.ComplexWarning)
                output = gv.gvar({p_ss : self.data[p_ss][n, :] for p_ss in self.data}, self.covariance_frozen)

            return output
            
        if means_only:
            as_gvar = False
        elif as_gvar:
            means_only = False
        else:
            as_gvar = True

        temp = {}
        for p_ss in self.data:
            if n == 0:
                data = self.data[p_ss]
                if self.rw_factors is not None:
                    rwfs = self.rw_factors
            else:
                data = self.data[p_ss][self.bs_list[n-1]]
                if self.rw_factors is not None:
                    rwfs = self.rw_factors[self.bs_list[n-1]]
            
            temp[p_ss] = data
            if self.rw_factors is not None:
                temp[p_ss] = np.einsum('ij,i->ij', data, rwfs)

        if self.rw_factors is not None:
            temp['rw'] = rwfs

        if means_only:
            if self.rw_factors is None:
                return {p_ss : np.mean(temp[p_ss], axis=0) for p_ss in self.data}
            else:
                return {p_ss : np.mean(temp[p_ss], axis=0)/ np.mean(rwfs) for p_ss in self.data}
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp = gv.dataset.avg_data(temp)
            if self.rw_factors is None:
                return {p_ss : temp[p_ss] for p_ss in self.data}
            else:
                return {p_ss : temp[p_ss] / temp['rw'] for p_ss in self.data}


class JackknifedData(object):
    def __init__(self, data, rw_factors=None, preprocessed=False, decorrelate_keys=True, seed=None, n_copies=None):
        '''
        Object for generating or managing jackknife resamples
        Args:
            data: measurements (either raw correlators or resampled data)
            rw_factors: reweighting measurements
            preprocessed: specify whether data contains arrays of jackknife resamples
            decorrelate_keys: specify whether correlations between particles/src_snk should be kept or discarded.
                If there are many different particle/src_snk combinations, this should probably
                be set to false in order to prevent the covariance matrix from becoming singular
        '''

        if preprocessed:
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=np.ComplexWarning)

                data_temp = {}
                for p_ss in data:
                    mean_rs = np.mean(data[p_ss][1:], axis=0)
                    data_temp[p_ss] = (data[p_ss][1:] - mean_rs) *np.sqrt(len(data[p_ss][1:])-1) + mean_rs

                if decorrelate_keys:
                    covariance_frozen = gv.evalcov({p_ss : gv.dataset.avg_data(data_temp[p_ss], spread=True) for p_ss in data}) 
                else:
                    covariance_frozen = gv.evalcov(gv.dataset.avg_data(data_temp, spread=True)) 

            self.n_copies = data[list(data)[0]].shape[0] - 1
        else:
            covariance_frozen = None
            self.n_copies = data[list(data)[0]].shape[0] 

        self.covariance_frozen = covariance_frozen

        self.data = data
        self.rw_factors = rw_factors
        self.preprocessed = preprocessed
        #block_size = 1 # single-elimination jackknife
        #self.block_size = block_size
        #self.n_blocks = int(data[list(data)[0]].shape[0] / block_size)

    
    def resample(self, means_only=False, as_gvar=False):
        for n in tqdm.tqdm(range(self.n_copies+1), desc='Jackknifing: '):
            yield self._jackknife(n=n, means_only=means_only, as_gvar=as_gvar)


    def save_resamples(self):
        output = {}
        for p_ss in self.data:
            shape = [self.n_copies+ 1] + list(self.data[p_ss].shape[1:])
            output[p_ss] = np.empty(shape)

        for j, data_rs in enumerate(self.resample(means_only=True)):
            for p_ss in data_rs:
                output[p_ss][j] = data_rs[p_ss]

        return output

    
    def _jackknife(self, n, means_only=False, as_gvar=False):
        '''
        Note that means_only==True iff as_gvar==False
        '''
        if self.preprocessed:
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=np.ComplexWarning)
                # output = {p_ss: gv.gvar(self.data[p_ss][n, :], self.covariance_frozen[p_ss]) 
                #     for p_ss in self.data}
                output = gv.gvar({p_ss : self.data[p_ss][n, :] for p_ss in self.data}, self.covariance_frozen)

                return output
            
        if means_only:
            as_gvar = False
        elif as_gvar:
            means_only = False
        else:
            as_gvar = True

        temp = {}
        for p_ss in self.data:
            if n == 0:
                data = self.data[p_ss]
                if self.rw_factors is not None:
                    rwfs = self.rw_factors
            else:
                data = np.delete(self.data[p_ss], n-1, axis=0)
                if self.rw_factors is not None:
                    rwfs = np.delete(self.rw_factors, n-1)

            temp[p_ss] = data
            if self.rw_factors is not None:
                temp[p_ss] = np.einsum('ij,i->ij', data, rwfs)

        if self.rw_factors is not None:
            temp['rw'] = rwfs

        if means_only:
            if self.rw_factors is None:
                return {p_ss : np.mean(temp[p_ss], axis=0) for p_ss in self.data}
            else:
                return {p_ss : np.mean(temp[p_ss], axis=0)/ np.mean(rwfs) for p_ss in self.data}
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp = gv.dataset.avg_data(temp)
            if self.rw_factors is None:
                return {p_ss : temp[p_ss] for p_ss in self.data}
            else:
                return {p_ss : temp[p_ss] / temp['rw'] for p_ss in self.data}


class FitResampler(abc.ABC):
    def __init__(self, data, prior, fargs_unfmt, model_avg_args=None,
                 seed=None, n_copies=None, jackknife=True, preprocessed=False, rw_factors=None): 
        
        if jackknife:
            self.resampled_data = JackknifedData(
                data=data, n_copies=n_copies, preprocessed=preprocessed, rw_factors=rw_factors)
            self.rng = RNG(seed)
        else:
            self.resampled_data = BootstrappedData(
                data=data, seed=seed, n_copies=n_copies, preprocessed=preprocessed, rw_factors=rw_factors)
            self.rng = RNG(seed)

        self._dumped_prior = gv.dumps(prior)
        self._posterior = None
        self._fargs_unfmt = fargs_unfmt
        self.preprocessed = preprocessed
        self.jackknife = jackknife
        self.model_avg_args = model_avg_args


    @functools.cached_property
    @abc.abstractmethod
    def fit_args(self):
        pass


    @property
    def fit_args_mdl_avg(self):
        return self.fit_args.cartesian_product(**self.model_avg_args)
    

    @abc.abstractmethod
    def make_fitter(self, data=None, prior=None, p0=None, fit_args=None, **kwargs):
        pass


    @property
    def prior(self):
        # rebuild prior from representation _prior:
        # necessary for performance speed-up from switch_gvar/restore_gvar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gv.loads(self._dumped_prior)
    
    
    @property
    def posterior(self):
        if self._posterior is None:
            fit = self.make_fitter().fit
            self._posterior = gv.dumps(fit.p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gv.loads(self._posterior)
        

    def choose_random_idx(self, fits):

        weights = lsqfitics.calculate_weights(fits, ic='BAIC')
        index = lsqfitics.argsort(fits, ic='BAIC')

        r = np.random.uniform(0, 1)
        total_weight = 1
        for idx, w in zip(index, weights[index]):
            total_weight -= w
            if r > total_weight:
                return idx
            
        return idx
    

    def estimate_systematics(self, params='*', average=None):
        if self.model_avg_args is None:
            average = False
        elif average is None:
            average = True

        if isinstance(params, str):
            params = [params]

        if average:
            posteriors = [fitter.posterior for fitter in self.make_fitters_for_mdl_avg()]
        else:
            posteriors = [self.make_fitter().posterior]

        matched_params = []
        for p in params:
            for part in posteriors[0]:
                matched_params.extend(fnmatch.filter(list(posteriors[0][part]), p))
        params = np.unique(matched_params)

        # initialize output dict
        sys_err = {}

        for part in posteriors[0]:
            sys_err[part] = {}
            for p in posteriors:
                for key in params:
                    sys_err[part][key] = np.squeeze(np.zeros(list(p[part][key].shape)))

        fitters = self.make_fitters_for_mdl_avg()

        weights = lsqfitics.calculate_weights([f.fit for f in fitters], ic='BAIC')
        #output['weights'][j] = weights
        for part in posteriors[0]:
            for param in sys_err[part]:
                vals = np.array([f.posterior[part][param] for f in fitters])                
                mdl_unc = np.sqrt(
                    np.sum([gv.mean(v)**2*w for v, w in zip(vals, weights)], axis=0) 
                    - np.sum([gv.mean(v)*w for v, w in zip(vals, weights)], axis=0)**2)
                sys_err[part][param] = mdl_unc

        return sys_err


    def make_fitters_for_mdl_avg(self, data=None, prior=None, p0=None, **kwargs):
        output = [self.make_fitter(data=data, prior=prior, p0=p0, fit_args=fargs, **kwargs)
            for fargs in self.fit_args_mdl_avg]
        return output


    def resample(self, params='*', randomize_prior=False, use_prior_for_p0=True, average=None):
        
        if self.model_avg_args is None:
            average = False
        elif average is None:
            average = True

        if isinstance(params, str):
            params = [params]

        if average:
            posteriors = [fitter.posterior for fitter in self.make_fitters_for_mdl_avg()]
        else:
            posteriors = [self.make_fitter().posterior]

        matched_params = []
        for p in params:
            for part in posteriors[0]:
                matched_params.extend(fnmatch.filter(list(posteriors[0][part]), p))
        params = np.unique(matched_params)

        # initialize output dict
        output = {}
        #sys_err = {}
        # n_models = 1
        # if average:
        #     n_models = len(self.fit_args_mdl_avg)
        #     output['weights'] = np.zeros([self.resampled_data.n_copies+1] + [n_models])

        for part in posteriors[0]:
            output[part] = {}
            #sys_err[part] = {}
            for p in posteriors:
                for key in params:
                    #sys_err[part][key] = np.squeeze(np.zeros(list(p[part][key].shape)))
                    output[part][key] = np.squeeze(np.zeros(
                        [self.resampled_data.n_copies+1] + list(p[part][key].shape)))
                        # [self.resampled_data.n_copies+1] + [n_models] + list(p[part][key].shape)))


        p0 = None
        if use_prior_for_p0:
            p0 = gv.mean(self.prior) 

        # models_chosen = []

        if average:
            fitters = self.make_fitters_for_mdl_avg()
            weights = lsqfitics.calculate_weights([f.fit for f in fitters], ic='BAIC')

        gv.switch_gvar() 
        for j, r in enumerate(self.resampled_data.resample(as_gvar=False)):
            if j == 0 or not randomize_prior:
                prior = self.prior
            else:
                prior = self.randomize_prior()
                
            if average:
                if j == 0:
                    for part in posteriors[0]:
                        for param in output[part]:
                            vals = np.array([f.posterior[part][param] for f in fitters])
                            #output[part][param][j] = gv.mean(vals)
                            output[part][param][j] = gv.mean(lsqfitics.calculate_average(
                                vals, weights=weights))   

                else:
                    fitters = self.make_fitters_for_mdl_avg(data=r, prior=prior, p0=p0)

                    idx = lsqfitics.choose_random_index(weights=weights)
                    fitter = fitters[idx]
                    # models_chosen.append(idx)

                    posterior = fitter.posterior
                    for part in posterior:
                        for key in params:
                            # output[part][key][j] = (
                            #     gv.mean(posterior[part][key]))
                            output[part][key][j] = (
                                gv.mean(posterior[part][key]) 
                                - gv.mean(posteriors[idx][part][key])
                                + output[part][key][0])

            # old way of doing the average
            elif False:# and j != 0:
                fitters = self.make_fitters_for_mdl_avg(data=r, prior=prior, p0=p0)

                weights = lsqfitics.calculate_weights([f.fit for f in fitters], ic='BAIC')
                #output['weights'][j] = weights
                for part in posteriors[0]:
                    for param in output[part]:
                        vals = np.array([f.posterior[part][param] for f in fitters])
                        #output[part][param][j] = gv.mean(vals)
                        output[part][param][j] = gv.mean(lsqfitics.calculate_average(
                            vals, weights=weights))
            else:
                fitter = self.make_fitter(data=r, prior=prior, p0=p0)
                posterior = fitter.posterior
                for part in output:
                    for key in params:
                        output[part][key][j] = gv.mean(posterior[part][key])



            gv.restore_gvar()
            gv.switch_gvar() 

        return output


        # don't forget to squeeze!

        # if average:
        #     # compute \braket{\hat X}_M^\text{r} - \braket{\hat X}_M +\braket{X}
        #     for part in output:
        #         for key in output[part]:
        #             # temp = output[part][key]
        #             unique_keys = np.unique(models_chosen[1:])
        #             for mdl_idx in unique_keys:
        #                 indicies = [j+1 for j, k in enumerate(models_chosen[1:]) if k == mdl_idx]
        #                 avg = np.mean(output[part][key][indicies], axis=0)
        #                 output[part][key][indicies] += -avg + output[part][key][0]
            

        # if average:
        #     def choose_random_idx(weights):
        #         index = np.argsort(weights)
        #         r = np.random.uniform(0, 1)
        #         total_weight = 1
        #         for idx, w in zip(index, weights[index]):
        #             total_weight -= w
        #             if r > total_weight:
        #                 return idx
                    
        #         return idx
            
        #     # compute \braket{\hat X}_M^\text{r} - \braket{\hat X}_M +\braket{X}
        #     averages = {}
        #     for part in posteriors[0]:
        #         averages[part] = {}
        #         for param in output[part]:
        #             averages[part][param] = {}
        #             for mdl_idx in range((output['weights']).shape[1]):
        #                 # indicies = [j+1 for j, k in enumerate(models_chosen[1:]) if k == mdl_idx]
        #                 avg = np.mean(output[part][param][:, mdl_idx])
        #                 averages[part][param][mdl_idx] = avg
        #                 # output[part][key][indicies] += -avg + output[part][key][0]

        #     temp = {}
        #     for n in range(self.resampled_data.n_copies + 1):
        #         if n == 0:
        #             for part in posteriors[0]:
        #                 temp[part] = {}
        #                 for param in output[part]:
        #                     temp[param] = {}
        #                     temp[part][param] = [gv.mean(lsqfitics.calculate_average(
        #                         output[part][param][0], weights=output['weights'][0]
        #                     ))]
        #         else:
        #             for part in posteriors[0]:
        #                 for param in output[part]:
        #                     idx = choose_random_idx(output['weights'][0])
        #                     #print(idx)
        #                     #temp[part][param].append(output[part][param][n, idx] - averages[part][param][idx] + temp[part][param][0])
        #                     temp[part][param].append(output[part][param][n, idx] - output[part][param][0, idx] + temp[part][param][0])

        #     return temp
        # else:
        #     return output


    def plot_health(self, param, bs_N=None, randomize_prior=False, use_prior_for_p0=True, n_copies=None):
        if bs_N is None:
            # Include bs 0 and bootstrap bs_N = n_copies times
            bs_N = self.resampled_data.data[list(self.resampled_data.data)[0]].shape[0] + 1
        
        self.rng.reset()
        
        #if self.randomize_prior:
        #    self.reset_rng()

        p0 = None
        if use_prior_for_p0:
            p0 = gv.mean(self.prior)
            #p0 = gv.mean(self.make_fit(n=0, randomize_prior=False).prior) 

        chi2_list = np.empty((bs_N, 2))
        param_list = np.empty((bs_N, 2))
        #fit_str_list = []

        q = multiprocess.Queue()
        gv.switch_gvar() 
        for j, r in enumerate(self.resampled_data.resample(as_gvar=False)):
            if j == 0 or not randomize_prior:
                prior = self.prior
            else:
                prior = self.randomize_prior()

            # Need to use multiprocessor to efficiently manage memory
            process = lambda r, q : q.put(self.make_fitter(data=r, prior=prior, p0=p0))
            p = multiprocess.Process(target=process, args=(r, q))
            p.start()

            fitter = q.get()
            fit = fitter.fit
            ratio = fit.chi2 / fit.evalchi2(self.posterior)
            chi2_list[j] = np.array([gv.mean(ratio), gv.sdev(ratio)])
            param_list[j] = np.array([gv.mean(fit.p[param]), gv.sdev(fit.p[param])])

            p.join()
            
            #gv.dataset.avg_data(r)
            gv.restore_gvar()
            gv.switch_gvar() 

        '''
        gv.switch_gvar() 
        for n, resampled_corr in enumerate(self.resampled_data.resample()):
            if n == 0 or not randomize_prior:
                prior = self.prior
            else:
                prior = self.randomize_prior()

            fit = self.make_fitter(data=resampled_corr, prior=prior, p0=p0).fit
            ratio = fit.chi2 / fit.evalchi2(self.posterior)
            chi2_list[n] = np.array([gv.mean(ratio), gv.sdev(ratio)])
            param_list[n] = np.array([gv.mean(fit.p[param]), gv.sdev(fit.p[param])])

            gv.restore_gvar()
            gv.switch_gvar() 
        gv.restore_gvar()
        '''

        to_gvar = lambda arr : gv.gvar(arr[:, 0], arr[:, 1])

        param_list = to_gvar(param_list)
        chi2_list = to_gvar(chi2_list)
        fig, (ax_p, ax_chi2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 1], 'wspace':0, 'hspace':0.1})

        g = gv.dataset.avg_data(gv.mean(param_list[1:]), bstrap=True)
        ax_p.errorbar(x=range(bs_N), xerr=None, y=gv.mean(param_list), yerr=gv.sdev(param_list), 
            ls='',  label=str(g)+' [all]')
        ax_p.axhspan(gv.mean(param_list[0]) - gv.sdev(param_list[0]), gv.mean(param_list[0]) + gv.sdev(param_list[0]), 
            alpha=0.5, label=str(param_list[0])+' [bs_0]')
        ax_p.set_ylabel('p[%s]'%param)
        ax_p.set_ylim(gv.mean(g) - 10 *gv.sdev(g), gv.mean(g) + 10 *gv.sdev(g))

        ax_chi2.errorbar(x=range(bs_N), xerr=None, y=gv.mean(chi2_list), yerr=gv.sdev(chi2_list), ls='', mec='white')
        ax_chi2.axhline(1, ls='--', alpha=0.5)
        ax_chi2.set_ylabel(r'$\chi^2(p^*)/\chi^2(p_0)$')
        ax_chi2.set_yscale('log')
    
        # Highlight fits with discrepencies
        idx = np.greater(gv.mean(chi2_list)-gv.sdev(chi2_list), 1)
        g = gv.dataset.avg_data(gv.mean(param_list[(~idx)][1:]), bstrap=True)

        ax_p.errorbar(x=np.array(range(bs_N))[idx], xerr=None, y=gv.mean(param_list)[idx], yerr=gv.sdev(param_list)[idx], 
            ls='', color='red', label=str(g)+' [excl]')
        ax_chi2.errorbar(x=np.array(range(bs_N))[idx], xerr=None, y=gv.mean(chi2_list)[idx], yerr=gv.sdev(chi2_list)[idx], 
            ls='', marker='.', color='red')
        
        ax_p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_chi2.set_xlabel('$N$')
        
        plt.close()
        return fig


    def plot_histogram(self, data, width=None, xlabel=None):
        hist_data = data[1:]
        center = data[0]
        mean_rs = np.mean(data[1:], axis=0) 

        # correct for bias/spread
        if self.jackknife:
            hist_data = (hist_data - mean_rs) *np.sqrt(len(data[1:]) - 1) + mean_rs
        else:
            hist_data = (hist_data - mean_rs + center)

        fig, ax = plt.subplots()
        _, bins, _ = ax.hist(hist_data, bins=int(len(data[1:]) / 10) + 1, density=True, facecolor='mediumslateblue', alpha=1, zorder=1)

        g = corrfit.io.to_gvar(data, preprocessed=True, jackknife=self.jackknife, bootstrap=(not self.jackknife))
        mu = g.mean
        sigma = g.sdev

        # Overlay gaussian from bs0
        if width is not None:
            y = scipy.stats.norm.pdf(bins, data[0], width)
            ax.plot(bins, y, color='deeppink', linewidth=2, label='Gaussian:    '+ str(gv.gvar(data[0], width)), zorder=2)
            ax.fill_between(bins, y, color='deeppink', alpha=0.2, zorder=0)

        # Overlay Gaussian from bs data
        y = scipy.stats.norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, color='deepskyblue', linewidth=2, label='Resampled: '+ (str(g)), zorder=2)
        ax.fill_between(bins, y, color='deepskyblue', alpha=0.2, zorder=0)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        ax.set_yticks([])
        ax.legend()
        plt.close()

        return fig


    def plot_qq(self, data):
        fig, ax = plt.subplots()
        scipy.stats.probplot(data, dist='norm', plot=ax)
        plt.close()
        return fig


    def randomize_prior(self):
        def randomize_central_value(m, s, key):
            if self.jackknife:
                s = s / np.sqrt(self.resampled_data.n_copies)
            #s = s/3
            if key.startswith('E'):
                return self.rng.normal(m, scale=s)
            elif key.startswith('log('):
                return self.rng.lognormal(m, sigma=s)
                #return m
            else: 
                return self.rng.normal(m, scale=s)
                #return m
            # elif key.startswith('dE'):
            #     return np.abs(self.rng.normal(m, scale=s))
            #     #return m
            # else:
            #     return self.rng.normal(m, scale=s)
            
        output_prior = gv.BufferDict()
        for key in self.prior:
            means = gv.mean(self.prior[key])
            sigmas = gv.sdev(self.prior[key])

            if hasattr(self.prior[key], '__len__'):
                output_prior[key] = [gv.gvar(randomize_central_value(m, s, key), s) for (m, s) in zip(means, sigmas)]  
            else:
                output_prior[key] = gv.gvar(randomize_central_value(means, sigmas, key), sigmas)

        return output_prior