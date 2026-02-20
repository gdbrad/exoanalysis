import numpy as np 

class JackUtils:

    def create_jack_ens(raw_corr):
        Ncfgs = raw_corr.shape[0]
        avg =np.mean(raw_corr,axis=0)
        jacked = []
        for cfg in range(Ncfgs):

            tmp = (Ncfgs/(Ncfgs-1))*avg-(1/(Ncfgs-1))*raw_corr[cfg]

            jacked=np.append(jacked,tmp)
        return jacked.reshape(raw_corr.shape)

    def undo_jack_bias(jacked_corr):
    #### undo jackknife bias ################

      Ncfgs = len(jacked_corr)
      avg = np.mean(jacked_corr)
      unjacked=[]
      for cfg in range(Ncfgs):

        tmp = Ncfgs*avg-(Ncfgs-1)*jacked_corr[cfg]
        unjacked = np.append(unjacked,tmp)
      return unjacked
    
    
from random import randint
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import hashlib
import tqdm
import lsqfit
from scipy import stats

import corrfit.io
import corrfit.fitters


class BootstrappedCorrelator(object):
    def __init__(self, raw_correlators, seed=None, n_copies=None):
        if seed is None:
            seed = str(np.random.randint(1e6))

        n_configs = raw_correlators[list(raw_correlators)[0]].shape[0]
        if n_copies is None:
            n_copies = n_configs

        self.n_configs = n_configs
        self.n_copies = n_copies
        self.raw_correlators = raw_correlators
        self.seed = seed
        self.rng = RNG(seed=seed)
        self._bs_list = None


    def resample(self, means_only=False, as_gvar=True):
        self.rng.reset()
        for n in tqdm.tqdm(range(self.n_copies+1), desc='Bootstrapping: '):
            yield self._bootstrap(n=n, means_only=means_only, as_gvar=as_gvar)

    
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


    def _bootstrap(self, n, means_only=False, as_gvar=True):
        '''Note that means_only=True overrides as_gvar=True'''
        if means_only:
            as_gvar = False

        if n == 0:
            if means_only:
                output = {part_src_snk : np.mean(self.raw_correlators[part_src_snk], axis=0) for part_src_snk in self.raw_correlators}
            else:
                output = self.raw_correlators

        else:
            output = {}
            for part_src_snk in self.raw_correlators:
                temp = self.raw_correlators[part_src_snk][self.bs_list[n-1], :]
                mean_bs = np.mean(temp, axis=0)
                if means_only:
                    output[part_src_snk] = mean_bs
                elif as_gvar:
                    output[part_src_snk] = temp

        if as_gvar:
            return gv.dataset.avg_data(output, bstrap=False)
        else:
            return output


class JackknifedCorrelator(object):
    def __init__(self, raw_correlators, seed=None):
        '''
        if block_size is None:
            block_size = 1 # single-elimination jackknife
        elif block_size % raw_correlators[list(raw_correlators)[0]].shape[0] != 0:
            print('Warning: block size not divisible by n_configs')
        '''

        block_size = 1 # single-elimination jackknife
            
        self.raw_correlators = raw_correlators
        self.block_size = block_size
        self.n_blocks = int(raw_correlators[list(raw_correlators)[0]].shape[0] / block_size)

    
    @property
    def n_copies(self):
        return self.n_blocks

    
    def resample(self):
        for n in tqdm.tqdm(range(self.n_blocks+1), desc='Jackknifing: '):
            yield self._jackknife(n=n)

    
    def _jackknife(self, n, means_only=False, as_gvar=True):
        if means_only:
            as_gvar = False

        if n == 0:
            if means_only:
                output = {part_src_snk : np.mean(self.raw_correlators[part_src_snk], axis=0) for part_src_snk in self.raw_correlators}
            else:
                output = self.raw_correlators

        else:
            output = {}
            for part_src_snk in self.raw_correlators:
                temp = np.delete(self.raw_correlators[part_src_snk], n-1, axis=0)
                if means_only:
                    output[part_src_snk] = np.mean(temp, axis=0)
                else:
                    output[part_src_snk] = temp

        if as_gvar:
            return gv.dataset.avg_data(output, bstrap=False)
        else:
            return output


class FitResampler(object):
    def __init__(self, raw_correlators, prior, fit_args, seed=None, n_copies=None, jackknife=False): 
        self.fit_args = corrfit.two_pt.FitArgs(
            data=gv.dataset.avg_data(raw_correlators), 
            fit_args=fit_args)

        if jackknife:
            self.resampled_correlator = JackknifedCorrelator(raw_correlators=raw_correlators)
            self.rng = RNG(seed)

        else:
            if n_copies is None:
                n_copies = raw_correlators[list(raw_correlators)[0]].shape[0]
            self.resampled_correlator = BootstrappedCorrelator(raw_correlators=raw_correlators, seed=seed, n_copies=n_copies)
            self.rng = RNG(seed)

        self._dumped_prior = gv.dumps(prior)
        self._posterior = None


    @property
    def prior(self):
        # rebuild prior from representation _prior:
        # necessary for performance speed-up from switch_gvar/restore_gvar
        return gv.gvar(gv.loads(self._dumped_prior))

    
    @property
    def posterior(self):
        if self._posterior is None:
            fit = self.make_fitter(correlators=gv.dataset.avg_data(self.resampled_correlator.raw_correlators),
                                    prior=self.prior).fit
            self._posterior = gv.dumps(fit.p)
        return gv.gvar(gv.loads(self._posterior))


    def resample(self, randomize_prior=False, use_prior_for_p0=True):
        
        posterior = self.make_fitter(
            correlators=gv.dataset.avg_data(self.resampled_correlator.raw_correlators), 
            prior=self.prior
        ).posterior

        output = {}
        for part in posterior:
            output[part] = {}
            for key in posterior[part]:
                output[part][key] = np.zeros([self.resampled_correlator.n_copies+1] + list(posterior[part][key].shape))


        p0 = None
        if use_prior_for_p0:
            p0 = gv.mean(self.prior) 

        # Since we only need the mean values of the posterior,
        # we can discard the intermediate gvar.
        # This significantly reduces memory usage, thereby improving performance
        gv.switch_gvar() 
        for n, resampled_corr in enumerate(self.resampled_correlator.resample()):
            if n == 0 or not randomize_prior:
                prior = self.prior
            else:
                prior = self.randomize_prior()

            posterior = self.make_fitter(correlators=resampled_corr, prior=prior, p0=p0).posterior
            for part in posterior:
                for key in posterior[part]:
                    output[part][key][n] = gv.mean(posterior[part][key])

            gv.restore_gvar()
            gv.switch_gvar() 
        gv.restore_gvar()

        return output


    def make_fitter(self, correlators, prior, p0=None, build_prior=True):
        fitter = corrfit.fitters.TwoPtFitter(
            data=correlators, 
            prior=prior,
            fit_args=self.fit_args,
            p0=p0,
            build_prior=build_prior
        )

        return fitter


    def plot_health(self, param='E0', bs_N=None, randomize_prior=False, use_prior_for_p0=True, n_copies=None):
        if bs_N is None:
            # Include bs 0 and bootstrap bs_N = n_copies times
            bs_N = self.resampled_correlator.raw_correlators[list(self.resampled_correlator.raw_correlators)[0]].shape[0] + 1
        
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

        gv.switch_gvar() 
        for n, resampled_corr in enumerate(self.resampled_correlator.resample()):
            if n == 0 or not randomize_prior:
                prior = self.prior
            else:
                prior = self.randomize_prior()

            fit = self.make_fitter(correlators=resampled_corr, prior=prior, p0=p0).fit
            ratio = fit.chi2 / fit.evalchi2(self.posterior)
            chi2_list[n] = np.array([gv.mean(ratio), gv.sdev(ratio)])
            param_list[n] = np.array([gv.mean(fit.p[param]), gv.sdev(fit.p[param])])

            gv.restore_gvar()
            gv.switch_gvar() 
        gv.restore_gvar()

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
        ax_chi2.set_ylabel('$\chi^2(p^*)/\chi^2(p_0)$')
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


    def plot_histogram(self, data, bs0_width=None):

        bins = int(len(data[1:]) / 10)

        fig, ax = plt.subplots()
        _, bins, _ = plt.hist(data[1:], bins=bins, density=True, facecolor='green', alpha=0.75)

        #mu = np.median(data[1:])
        #sigma = np.std(data[1:])
        g = gv.dataset.avg_data(data[1:], bstrap=True)
        mu = g.mean
        sigma = g.sdev

        # Overlay gaussian from bs0
        if bs0_width is not None:
            y = stats.norm.pdf(bins, data[0], bs0_width)
            plt.plot(bins, y, 'b--', linewidth=1, label='bs0: %s'%(str(gv.gvar(data[0], bs0_width))))

        # Overlay Gaussian from bs data
        y = stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--', linewidth=1, label='bs:   %s'%(str(g)))

        plt.ylabel('Frequency')
        #plt.grid(True)
        plt.legend()

        #text = ('$\overline{p} (s_{\overline{p}}) = $ %s' %(gv.gvar(mu, sigma)))

        # these are matplotlib.patch.Patch properties
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        #ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
        #        verticalalignment='top', bbox=props)

        fig = plt.gcf()
        plt.close()

        return fig


    def plot_qq(self, data):
        fig, ax = plt.subplots()
        stats.probplot(data, dist='norm', plot=ax)
        plt.close()
        return fig


    def randomize_prior(self):
        def randomize_central_value(m, s, key):
            return m 
            if key == 'log(dE)':
                return self.rng.lognormal(m, sigma=s)
                #return m
            elif key == 'dE':
                return np.abs(self.rng.normal(m, scale=s))
                #return m
            else:
                return self.rng.normal(m, scale=s/3)
            
        output_prior = gv.BufferDict()
        for key in self.prior:
            means = gv.mean(self.prior[key])
            sigmas = gv.sdev(self.prior[key])

            if hasattr(self.prior[key], '__len__'):
                output_prior[key] = [gv.gvar(randomize_central_value(m, s, key), 20*s) for j, (m, s) in enumerate(zip(means, sigmas))]  
            else:
                output_prior[key] = gv.gvar(randomize_central_value(means, 20*sigmas, key), sigmas)

        return output_prior


    def estimate_priors(self, corr_data):
        # Need to fix this method
        return None
        #return self.prior

        output = {}
        prior = self.prior
        output.update(prior)

        effective_mass = {}
        for smr in corr_data:
            if self.particle_statistics == 'fermi-dirac':
                effective_mass[smr] = np.log(corr_data[smr] / np.roll(corr_data[smr] , -1))
            elif self.particle_statistics == 'bose-einstein':
                effective_mass[smr] = np.arccosh(
                (np.roll(corr_data[smr] , -1) + np.roll(corr_data[smr] , 1))
                    /(2*corr_data[smr] )
                )

        e0_estimate = lsqfit.wavg(np.concatenate([effective_mass[smr][10:15] for smr in effective_mass]))
        output['E0'] = gv.gvar(gv.mean(e0_estimate), gv.sdev(prior['E0']))

        effective_wf = {}
        for smr in corr_data:
            t = np.arange(len(corr_data[smr]))
            if self.particle_statistics == 'fermi-dirac':
                effective_wf[smr] = np.exp(effective_mass[smr] *t) *corr_data[smr]
            elif self.particle_statistics == 'bose-einstein':
                if self.overlap == 'ZZ':
                    effective_wf[smr] = corr_data[smr] / (np.exp(-effective_mass[smr] *t) + np.exp(-effective_mass[smr] *(self.t_period - t)))
                else:
                    effective_wf[smr] = 1 / np.cosh(effective_mass[smr] *(t - self.t_period/2)) *corr_data[smr]

        wf_estimate = {smr : lsqfit.wavg(effective_wf[smr][10:12]) for smr in effective_wf}
        if self.overlap == 'ZZ':
            wf_smr = np.sqrt(gv.mean(wf_estimate['smr']))
            wf_dir = gv.mean(wf_estimate['dir']/ np.sqrt(wf_estimate['smr']))
            output['Z_smr'] = gv.gvar(wf_smr, wf_smr)
            output['Z_dir'] = gv.gvar(wf_dir, wf_dir)

        return output