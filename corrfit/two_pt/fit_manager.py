import warnings
import numpy as np
import gvar as gv
import tqdm
import matplotlib.pyplot as plt
import matplotlib

import lsqfitics
import corrfit.base
from corrfit.plot import default_cmap
from corrfit.utils import fmt_tuple_as_str, pm


class FitManager(corrfit.base.FitManager):

    def __init__(self, correlators, prior=None, fargs_unfmt=None, 
            auto_prior_args=None, auto_fit_args=None):

        # Default values
        self.correlators = correlators
        self.auto_fit_args = None

        self._effective_mass = None
        self._effective_wf = None
        self._effective_z = None
        self._avg_E0 = None
        self._fitters = None

        self.part_src_snks_list = list(correlators)

        if fargs_unfmt is None:
            fargs_unfmt = {}
        for part, _ in self.part_src_snks_list:
            if (part not in fargs_unfmt) or fargs_unfmt[part] is None:
                fargs_unfmt[part] = {}

        if auto_fit_args is not None:
            if 'overlap' in auto_fit_args:
                for p_ss in fargs_unfmt:
                    fargs_unfmt[p_ss]['overlap'] = auto_fit_args['overlap']
        self._fargs_unfmt = fargs_unfmt 

        if (auto_prior_args is not None and auto_prior_args['enabled']):
            raise NotImplementedError
            auto_prior_args.setdefault('tmin', None)
            auto_prior_args.setdefault('tmax', None)
            auto_prior_args.setdefault('nsamples', 400)
            auto_prior_args.setdefault('mpi', None)
            auto_prior_args.setdefault('a', None)
            auto_prior_args.setdefault('wide_prior', True)
            
            self.prior = self.estimate_prior(
                tmin=auto_prior_args['tmin'],
                tmax=auto_prior_args['tmax'],
                nsamples=auto_prior_args['nsamples'],
                mpi=auto_prior_args['mpi'],
                a=auto_prior_args['a'],
                wide_prior=auto_prior_args['wide_prior']
                )
        else:
            self.prior = prior
        
        # Will override fit_args if present
        if auto_fit_args is not None and 'enabled' in auto_fit_args:
            raise NotImplementedError
            auto_fit_args.setdefault('t_start', 3)
            fit_args = self.get_highest_weight_fit_args(
                t_start=auto_fit_args['t_start'],
                t_end=auto_fit_args['t_end'],
                n_max=auto_fit_args['n_max']
            )

            self.auto_fit_args = auto_fit_args
    

    # as properties so they can be easily overwritten
    @property
    def fit_args(self):
        return corrfit.two_pt.fit_args.FitArgs(fargs_unfmt=self._fargs_unfmt, data_keys=list(self.correlators))


    @property
    def fitters(self):
        if self._fitters is None:
            self._fitters = corrfit.two_pt.fitters_dict.FittersDict(
                data=self.correlators, prior=self.prior)
        return self._fitters
    

    # def __str__(self):
    #     fit_args = self.fit_args
    #     output = 'Fit args:\n'
    #     max_len = np.max([len(str(p))+len(str(ss)) for p, ss in fit_args])
    #     for p, ss in fit_args:
    #         output += (str(p)+' ['+str(ss)+']:').rjust(max_len+7)+'   '
    #         if self.fit_args[p, ss]['perform_fit']:
    #             output += ('n_states = %s    '%(fit_args[p, ss]['n_states'])).ljust(13)
    #             output += 't = [%s, %s)\n'%(fit_args[p, ss]['t_start'], fit_args[p, ss]['t_end'])
    #         else:
    #             output += 'No fit performed\n'
    #     output += '\n'
    #     output += str(self.get_fits())
    #     return output


    @property
    def effective_mass(self):
        if self._effective_mass is None:
            # Ignore irrelevant log warnings, 
            # which only occur for times with unuseable SNRs
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                effective_mass = {}
                for part, src_snk in self.correlators:
                    if self._get_particle_statistics(part, src_snk) == 'fermi-dirac':
                        effective_mass[(part, src_snk)] = np.log(self.correlators[(part, src_snk)] / np.roll(self.correlators[(part, src_snk)] , -1))
                    elif self._get_particle_statistics(part, src_snk) == 'bose-einstein':
                            effective_mass[src_snk] = np.arccosh(
                            (np.roll(self.correlators[(part, src_snk)] , -1) + np.roll(self.correlators[(part, src_snk)] , 1))
                                /(2*self.correlators[(part, src_snk)] )
                            )
            self._effective_mass = effective_mass
        return self._effective_mass


    @property
    def effective_wf(self):
        if self._effective_wf is None:
            effective_wf = {}
            for part, src_snk in self.correlators:
                t = np.arange(len(self.correlators[(part, src_snk)]))
                if self._get_particle_statistics(part, src_snk) == 'fermi-dirac':
                    effective_wf[(part, src_snk)] = np.exp(self.effective_mass[(part, src_snk)] *t) *self.correlators[(part, src_snk)]
                elif self._get_particle_statistics(part, src_snk) == 'bose-einstein':
                    if self.overlap in ['ZZ', 'matrix']:
                        effective_wf[(part, src_snk)] = (
                            self.correlators[(part, src_snk)] / 
                            (np.exp(-self.effective_mass[(part, src_snk)] *t) 
                             + np.exp(-self.effective_mass[(part, src_snk)] *(self._get_t_period(part) - t))))
                    else:
                        effective_wf[(part, src_snk)] = 1 / np.cosh(self.effective_mass[(part, src_snk)] *(t - self._get_t_period(part)/2)) *self.correlators[(part, src_snk)]
            self._effective_wf = effective_wf
        return self._effective_wf


    @property
    def effective_z(self):
        if self._effective_z is None:
            effective_z = {}
            for part, (src, snk) in self.correlators:
                if self.fit_args.get((part, (src, snk)))['overlap'] not in 'ZZ' or self._get_particle_statistics(part, (src, snk)) == 'bose-einstein':
                    effective_z[(part, src)] = [None] *len(self.correlators[(part, (src, snk))])
                elif src == snk:
                    t = np.arange(len(self.correlators[(part, (src, snk))]))
                    effective_z[(part, src)] = np.sqrt(self.correlators[(part, (src, snk))]) *np.exp(self.effective_mass[(part, (src, snk))] *t/2)

            for part, (src, snk) in self.correlators:
                if self.fit_args.get((part, (src, snk)))['overlap'] not in 'ZZ' or self._get_particle_statistics(part, (src, snk)) == 'bose-einstein':
                    pass
                elif src != snk and (part, src) not in effective_z:
                    try:
                        t = np.arange(len(self.correlators[(part, (src, snk))]))
                        effective_z[(part, src)] = (
                            self.correlators[(part, (src, snk))] 
                            *np.exp(self.effective_mass[(part, (src, snk))] *t)
                            / effective_z[(part, snk)])
                    except KeyError:
                        pass
                elif src != snk and (part, snk) not in effective_z:
                    try:
                        t = np.arange(len(self.correlators[(part, (src, snk))]))
                        effective_z[(part, snk)] = (
                            self.correlators[(part, (src, snk))] 
                            *np.exp(self.effective_mass[(part, (src, snk))] *t)
                            / effective_z[(part, src)])
                    except KeyError:
                        pass

            self._effective_z = effective_z
        return self._effective_z


    # decorator
    # used to iterate over multiple fit_args, 
    # e.g., kwargs = {t_start :  [5, 7]} -> kwargs_list = [{t_start : 5}, {t_start : 7}]
    # def iterate_fit_args(func):

    #     def inner(self, **kwargs):
    #         kwargs.setdefault('particles', None)
    #         fargs_keys = ['n_states', 't_start', 't_end', 'energy_gaps', 'svdcut', 'prior_En']
    #         for key in fargs_keys:
    #             kwargs.setdefault(key, None)

    #         duplicate_keys = {}
    #         for key in fargs_keys:
    #             if key != 'particles':
    #                 temp = kwargs[key]
    #                 if temp is not None and not isinstance(temp, str) and hasattr(temp, '__len__'):
    #                     duplicate_keys[key] = kwargs[key]

    #         if len(duplicate_keys) == 0:
    #             return func(self, **kwargs)

    #         kwargs_list = []
    #         for temp_prod in itertools.product(*duplicate_keys.values()):
    #             iter_dict = dict(zip(duplicate_keys.keys(), temp_prod))
    #             temp = {}
    #             temp.update(kwargs)
    #             temp.update(iter_dict)
    #             kwargs_list.append(temp)

    #         return [func(self, **k) for k in tqdm.tqdm(kwargs_list, desc='Collecting FitArgs: ')]

    #     return inner 


    #@iterate_fit_args

    # def get_fit_args(self, particles=None, t_start=None, t_end=None, n_states=None, energy_gaps=None, 
    #         svdcut=None, prior_En=None, defaults_only=False):
    #     if self.fit_args is None:
    #         raise NameError('fit_args unspecified when instantiating FitManager object')
        
    #     if particles is None:
    #         particles = sorted(list(self.fit_args))
    #     elif particles == 'all':
    #         particles = self._get_particles()
    #     elif isinstance(particles, str):
    #         particles = [particles] 

    #     # defaults = {}
    #     # defaults['t_start'] = np.nanmin([self.fit_args[part]['t_start'] 
    #     #     for part in particles if self.fit_args[part]['perform_fit']])
    #     # defaults['t_end'] = np.nanmax([self.fit_args[part]['t_end'] 
    #     #     for part in particles if self.fit_args[part]['perform_fit']])
    #     # defaults['n_states'] = np.nanmax([self.fit_args[part]['n_states'] 
    #     #     for part in particles if self.fit_args[part]['perform_fit']])
    #     # defaults['energy_gaps'] = 'constant'
    #     # if all([self.fit_args[part]['svdcut'] is None for part in self.fit_args if self.fit_args[part]['perform_fit']]):
    #     #     defaults['svdcut'] = None
    #     # else:
    #     #     defaults['svdcut'] = np.max([self.fit_args[part]['svdcut'] for part in self.fit_args 
    #     #         if self.fit_args[part]['perform_fit'] and self.fit_args[part]['svdcut'] is not None])
    #     # defaults['particles'] = particles
    #     # defaults['prior_En'] = False
    #     defaults = self.fit_args[particle, 'default']
            
    #     if defaults_only:
    #         return defaults

    #     fit_args = {}        
    #     for part in particles:
    #         if part in self.fit_args:
    #             fit_args[part] = copy.deepcopy(self.fit_args[part])
    #         else:
    #             fit_args[part] = copy.deepcopy(defaults)

    #         fit_args[part]['n_states']    = n_states    or fit_args[part]['n_states']
    #         fit_args[part]['t_start']     = t_start     or fit_args[part]['t_start']
    #         fit_args[part]['t_end']       = t_end       or fit_args[part]['t_end']
    #         fit_args[part]['energy_gaps'] = energy_gaps or fit_args[part]['energy_gaps']
    #         fit_args[part]['svdcut']      = svdcut      or fit_args[part]['svdcut']

    #         # Be careful with boolean variables
    #         if prior_En is not None:
    #             fit_args[part]['prior_En'] = prior_En 

    #     return corrfit.two_pt.fit_args.FitArgs(fit_args=fit_args)  


    def _get_models(self, particles=None, src_snk=None, **kwargs):
        fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
        models = self.fitters[fit_args].models
        output = {model.datatag : model for model in models} 
        output = {k : v for k, v in sorted(output.items())}
        return output
    

    def _get_particles(self):
        return sorted(np.unique([part for part, _ in self.correlators]))


    def _get_particle_statistics(self, particle, src_snk):
        return self.fit_args.get((particle, src_snk))['particle_statistics']
    

    def _get_t_period(self, particle):
        return len([self.correlators[(p, s)] for p, s in self.correlators if p == particle][0])
        

    def get_highest_weight_fit_args(self, t_start=None, t_end=None, n_max=None):
        t_start = t_start or self.auto_fit_args['t_start']
        t_end = t_end or self.auto_fit_args['t_end']
        n_max = n_max or self.auto_fit_args['n_max']

        fit_args = self.get_fit_args(t_start=list(range(t_start, t_end)), t_end=t_end, n_states=list(range(1, n_max+1)))
        fits = self.get_fits(t_start=list(range(t_start, t_end)), t_end=t_end, n_states=list(range(1, n_max+1)))

        weights = self._calculate_weights(fits, ic='BAIC')
        return fit_args[np.argmax(weights)]


    def get_avg_E0(self, t_start=None, t_end=None, n_max=None):
        t_start = t_start or self.auto_fit_args['t_start']
        t_end = t_end or self.auto_fit_args['t_end']
        n_max = n_max or self.auto_fit_args['n_max']

        fits = self.get_fits(
            t_start=list(range(t_start, t_end)), 
            t_end=t_end, 
            n_states=list(range(1, n_max+1))
        )
        weights = self._calculate_weights(fits, ic='BAIC')

        output = {}
        for part_src_snk in self.part_src_snks_list:
            output[part_src_snk] = self._calculate_average([f.p[self.get_p_keys()[part_src_snk]['E0']] for f in fits], weights=weights)

        return output
            

    def estimate_prior(self, tmin, tmax, nsamples=400, mpi=None, a=None, wide_prior=True):
        def avg_from_random_fits(values, tmin, tmax, nsamples):
            fits = []
            for j in range(nsamples):
                size = np.random.choice(list(range(1, tmax-tmin)))
                t = np.random.choice(list(range(tmin, tmax)), size=size, replace=False)
                
                def fcn(x, p):
                    return p *np.repeat(1, len(x))
                
                fit = lsqfitics.nonlinear_fit(data=(t, values[t]), fcn=fcn, prior=gv.gvar(0, 1), dcut=tmax-tmin-size)
                fits.append(fit)

            constants = np.array([fit.p.item() for fit in fits])
            weights = self._calculate_weights(fits, ic='BAIC')
            return self._calculate_average(constants, weights=weights)
        
        def estimate_gap(mpi, a):
            mpi_lat =  mpi*a / 197.3269804
            return gv.gvar(2 *mpi_lat, mpi_lat)
        
        def format_tuple_as_str(src_snk):
            if isinstance(src_snk, str):
                return src_snk
            else:
                return '('+src_snk[0]+','+src_snk[1]+')'
        
        prior = gv.BufferDict()
        param_E0 = {}
        for part, src_snk in self.effective_mass:
            val = avg_from_random_fits(self.effective_mass[(part, src_snk)], tmin=tmin, tmax=tmax, nsamples=nsamples)
            if part in param_E0:
                #print(param_E0[part], val)
                param_E0[part] = gv.gvar(gv.mean((param_E0[part]+val)/2), np.max(gv.sdev([param_E0[part], val])))
            else:
                param_E0[part] = val

            if wide_prior:
                prior['E0::'+part] = gv.gvar(gv.mean(param_E0[part]), 0.1)
            else:
                prior['E0::'+part] = param_E0[part]

        param_dE = {}
        for part, src_snk in self.effective_mass:
            if mpi is None or a is None:
                param_dE[part] = gv.gvar(gv.mean(param_E0[part]), gv.mean(param_E0[part]))
            else:
                param_dE[part] = estimate_gap(mpi, a)
            prior['dE::'+part] = param_dE[part]


        param_wf = {}
        for part, src_snk in self.effective_wf:
            if self.fit_args[part]['overlap'] != 'ZZ':
                val = avg_from_random_fits(self.effective_wf[(part, src_snk)], tmin=tmin, tmax=tmax, nsamples=nsamples)
                param_wf[(part, src_snk)] = val
                
                if wide_prior:
                    prior['wf::'+part+'::'+format_tuple_as_str(src_snk)] = gv.gvar(gv.mean(param_wf[(part, src_snk)]), gv.mean(param_wf[(part, src_snk)]))
                else:
                    prior['wf::'+part+'::'+format_tuple_as_str(src_snk)] = param_wf[(part, src_snk)]

        param_z = {}
        for part, smr in self.effective_z:
            if self.fit_args[part]['overlap'] == 'ZZ':
                val = avg_from_random_fits(self.effective_z[(part, smr)], tmin=tmin, tmax=tmax, nsamples=nsamples)
                param_z[(part, smr)] = val

                if wide_prior:
                    prior['Z::'+part+'::'+smr] = gv.gvar(gv.mean(param_z[(part, smr)]), gv.mean(param_z[(part, smr)]))
                else:
                    prior['Z::'+part+'::'+smr] = param_z[(part, smr)]

        return prior


    def get_p_keys(self, particles=None, src_snk=None, **kwargs):
        fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
        return self.fitters[fit_args].p_keys


    def get_spectrum(self, particles=None, src_snk=None, **kwargs):
        fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
        if isinstance(fit_args, list):
            return [self.fitters[fargs].spectrum for fargs 
                in tqdm.tqdm(fit_args, desc="Collecting fits: ")] 
        else:
            return self.fitters[fit_args].spectrum


    def fcn_correlators(self, t, particles=None, src_snk=None, **kwargs):
        p = self.get_fits(particles=particles, src_snk=src_snk, **kwargs).p
        output = {}
        for datatag, model in sorted(
                self._get_models(particles=particles, src_snk=src_snk, **kwargs).items()):
            if datatag[0] == 'two_pt':
                output[(datatag[1], datatag[2])] = model.fitfcn(p, t)
        return output


    def fcn_effective_mass(self, t, particles=None, src_snk=None, **kwargs):
        p = self.get_fits(particles=particles, src_snk=src_snk, **kwargs).p
        output = {}
        for datatag, model in sorted(
                self._get_models(particles=particles, src_snk=src_snk, **kwargs).items()):
            if datatag[0] == 'two_pt':
                output[(datatag[1], datatag[2])] = model.fcn_effective_mass(p, t)
        return output


    def fcn_effective_wf(self, t, particles=None, src_snk=None, **kwargs):
        p = self.get_fits(particles=particles, src_snk=src_snk, **kwargs).p
        output = {}
        for datatag, model in sorted(
                self._get_models(particles=particles, src_snk=src_snk, **kwargs).items()):
            if datatag[0] == 'two_pt':
                output[(datatag[1], datatag[2])] = model.fcn_effective_wf(p, t)
        return output


    def fcn_effective_z(self, t, particles=None, src_snk=None, **kwargs):
        corr_fit = self.fcn_correlators(t, particles=particles, src_snk=src_snk, **kwargs)
        eff_mass_fit = self.fcn_effective_mass(t, particles=particles, src_snk=src_snk, **kwargs)
        eff_wf_fit = self.fcn_effective_wf(t, particles=particles, src_snk=src_snk, **kwargs)

        if particles is None:
            particles = self._get_particles()

        output = {}
        for part, src_snk in sorted([(p, ss) for p, ss in self.correlators if p in particles]):
            if self.fit_args.get((part, src_snk))['overlap'] in ['ZZ'] and self._get_particle_statistics(part, src_snk) == 'fermi-dirac':
                src, snk = src_snk
    
                if src == snk:
                    output[part, src] = np.sqrt(corr_fit[(part, src_snk)]) *np.exp(eff_mass_fit[(part, src_snk)] *t/2)
                
        for part, src_snk in sorted([(p, ss) for p, ss in self.correlators if p in particles]):
            if (part, src_snk) in corr_fit:
                if self.fit_args.get((part, src_snk))['overlap'] in ['ZZ'] and self._get_particle_statistics(part, src_snk) == 'fermi-dirac':                
                    src, snk = src_snk
                    
                    if src != snk and (part, src) not in output:
                        output[part, src] = (
                            corr_fit[(part, src_snk)]
                            *np.exp(eff_mass_fit[(part, src_snk)] *t)
                            / output[part, snk])
                    if src != snk and (part, snk) not in output:
                        output[part, snk] = (
                            corr_fit[(part, src_snk)] 
                            *np.exp(eff_mass_fit[(part, src_snk)] *t)
                            / output[part, src])
                            
                elif self._get_particle_statistics(part, src_snk) == 'bose-einstein':
                    print('Warning: fcn_effective_z not yet implemented for bose-einstein statistics')
                    output[part, src_snk] = [np.nan] *len(self.correlators[part, src_snk])
                else:
                    output[part, src_snk] = np.sqrt(eff_wf_fit[part, src_snk])

        return output


    def _set_default_plot_lims(self, t_start=None, t_end=None, show_fit=False):
        t_period = np.nanmax([self._get_t_period(part) for part in [p for p, _ in self.correlators]])

        if t_start is None:
            temp = [self.fit_args.get(p_ss)['t_start'] for p_ss in self.correlators]
            if all(ti is None for ti in temp):
                t_start = 3
            else:
                t_start = np.nanmin(temp)
                
        if t_end is None:
            temp = [self.fit_args.get(p_ss)['t_end'] for p_ss in self.correlators]
            if all(ti is None for ti in temp):
                t_end = int(t_period/3)
            else:
                t_end = np.nanmin(temp)

        if all([self._get_particle_statistics(p, ss) == 'fermi-dirac' 
                for p, ss in self.correlators]
            ):
            if show_fit:
                t_plot_min = int(np.mean([2, t_start]))
            else:
                t_plot_min = int(t_period/16)

            if show_fit:
                t_plot_max = int((t_period + 2 *t_end)/3)
                t_plot_max = int(np.min([t_end *1.5, t_period]))
            else:
                t_plot_max = int(t_period/3)

        else: # if any particles obey bose-einstein statistics
            if show_fit:
                t_plot_min = int(np.mean([0, t_start]))
            else:
                t_plot_min = t_period/16

            if show_fit:
                t_plot_max = int((t_period + t_end)/2)
            else:
                t_plot_max = int(15 *t_period/16)

        return t_plot_min, t_plot_max



    ###############
    # plots below #
    ###############
    def plot_correlators(self, show_fit=True, t_plot_min=None, t_plot_max=None, ylim=None, show_legend=True, 
            particles=None, src_snk=None, **kwargs):
        
        if self.get_fits(particles=particles, src_snk=src_snk, **kwargs) is None:
            show_fit = False

        if t_plot_min is None:
            t_plot_min, _ = self._set_default_plot_lims(show_fit=show_fit)
        if t_plot_max is None:
            _, t_plot_max = self._set_default_plot_lims(show_fit=show_fit)

        if particles is None or particles == 'all':
            particles = self._get_particles()
        elif isinstance(particles, str):
            particles = [particles]

        if src_snk is None:
            correlators = {(p,ss) : self.correlators[(p, ss)] for p, ss in self.correlators if p in particles}
        else:
            correlators = {(p,ss) : self.correlators[(p, ss)] for p, ss in self.correlators if p in particles
                           and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)}

        fig, ax = plt.subplots()
        if show_fit:
            fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)

            t = np.linspace(t_plot_min, t_plot_max)
            corr_fit = self.fcn_correlators(t, particles=particles, src_snk=src_snk, **kwargs)

            for j, (part, src_snk) in enumerate(sorted(correlators)):
                if (part, src_snk) in corr_fit:

                    color = default_cmap((j+1)/(len(correlators)+1))

                    ax.plot(t, pm(corr_fit[(part, src_snk)], 0), '--', color=color)
                    ax.plot(t, pm(corr_fit[(part, src_snk)], 1), t, pm(corr_fit[(part, src_snk)], -1), color=color)
                    ax.fill_between(t, pm(corr_fit[(part, src_snk)], -1), pm(corr_fit[(part, src_snk)], 1),
                                    facecolor=color, alpha = 0.10, rasterized=True)

                    if fit_args.get((part, src_snk)) is not None:
                        ax.axvline(fit_args.get((part, src_snk))['t_start']-0.5, linestyle=(j, (1, len(correlators))), alpha=0.8, color=color)
                        ax.axvline(fit_args.get((part, src_snk))['t_end']-0.5, linestyle=(j, (1, len(correlators))), alpha=0.8, color=color)

        ax = self._plot_quantity(quantity=correlators, ax=ax,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel='$\log C(t)$', ylim=ylim, show_legend=show_legend)
        ax.set_yscale('log')
        plt.close()
        return fig


    def plot_effective_mass(self, show_fit=True, t_plot_min=None, t_plot_max=None, ylim=None, show_legend=True, show_all=False, 
            particles=None, src_snk=None, extra_bands={}, **kwargs):
        
        if self.get_fits(particles=particles, src_snk=src_snk, **kwargs) is None:
            show_fit = False

        if t_plot_min is None:
            t_plot_min, _ = self._set_default_plot_lims(show_fit=show_fit)
        if t_plot_max is None:
            _, t_plot_max = self._set_default_plot_lims(show_fit=show_fit)   

        if particles is None or particles == 'all':
            particles = self._get_particles()
        elif isinstance(particles, str):
            particles = [particles]

        if src_snk is None:
            effective_mass = {(p,ss) : self.effective_mass[(p, ss)] for p, ss in self.effective_mass if p in particles}
        else:
            effective_mass = {(p,ss) : self.effective_mass[(p, ss)] for p, ss in self.effective_mass if p in particles
                           and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)}
        
        fig, ax = plt.subplots()
        if show_fit:
            fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
            # create colormap from 'cool' and 'winter'

            t = np.linspace(t_plot_min, t_plot_max)
            effective_mass_fit = self.fcn_effective_mass(t, particles=particles, src_snk=src_snk, **kwargs)

            for j, (part, src_snk) in enumerate(sorted(effective_mass)):
                if (part, src_snk) in effective_mass_fit:

                    color = default_cmap((j+1)/(len(effective_mass)+1))

                    ax.plot(t, pm(effective_mass_fit[(part, src_snk)], 0), '--', color=color)
                    ax.plot(t, pm(effective_mass_fit[(part, src_snk)], 1), 
                                t, pm(effective_mass_fit[(part, src_snk)], -1), color=color)
                    ax.fill_between(t, pm(effective_mass_fit[(part, src_snk)], -1), pm(effective_mass_fit[(part, src_snk)], 1),
                                    facecolor=color, alpha = 0.10, rasterized=True)

                    if fit_args.get((part, src_snk)) is not None:
                        ax.axvline(fit_args.get((part, src_snk))['t_start']-0.5, linestyle=(j, (1, len(effective_mass))), alpha=0.8, color=color)
                        ax.axvline(fit_args.get((part, src_snk))['t_end']-0.5, linestyle=(j, (1, len(effective_mass))), alpha=0.8, color=color)

        if show_all:
            ylim = (None, None)

        elif (ylim is None and show_fit
            and all([True if (part, src_snk) in effective_mass_fit
                     else False for (part, src_snk) in effective_mass])):

            #spectrum = self.get_spectrum(particles=particles)
            #y_low = np.nanmin([pm(spectrum[key][0], -10) for key in spectrum])
            #y_high = np.nanmax([pm(spectrum[key][0], +10) for key in spectrum]) 
            E0_keys = [self.get_p_keys(particles=particles, src_snk=src_snk, **kwargs)[p_ss]['E0'] 
                       for p_ss in self.get_p_keys(particles=particles, src_snk=src_snk, **kwargs) if p_ss[0] in particles]
            y_low = np.nanmin([pm(self.prior[k], -0.5) for k in E0_keys])
            y_high = np.nanmax([pm(self.prior[k], 0.5) for k in E0_keys])
            ylim = (y_low, y_high)

        if extra_bands is not None:
            for j, (key, val) in enumerate(extra_bands.items()):
                cmap = matplotlib.colormaps['spring']
                color = cmap((j+1)/(len(extra_bands)+1))
                ax.axhspan(pm(val, -1), pm(val, 1), label=key, alpha=0.5, color=color)

        ax = self._plot_quantity(
            quantity=effective_mass, ax=ax,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel=r'$am_{\rm eff}(t)$', ylim=ylim, show_legend=show_legend)

        plt.close()
        return fig


    def plot_effective_wf(self, show_fit=True, show_all=False, t_plot_min=None, t_plot_max=None, show_legend=True, 
            particles=None, src_snk=None, **kwargs):

        if self.get_fits(particles=particles, src_snk=src_snk, **kwargs) is None:
            show_fit = False

        if t_plot_min is None:
            t_plot_min, _ = self._set_default_plot_lims(show_fit=show_fit)
        if t_plot_max is None:
            _, t_plot_max = self._set_default_plot_lims(show_fit=show_fit)  

        particles = kwargs.get('particles')
        if particles is None or particles == 'all':
            particles = self._get_particles()
        elif isinstance(particles, str):
            particles = [particles]
        
        if src_snk is None:
            effective_wf = {(p,ss) : self.effective_wf[(p, ss)] for p, ss in self.effective_wf if p in particles}
        else:
            effective_wf = {(p,ss) : self.effective_wf[(p, ss)] for p, ss in self.effective_wf 
                if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk) }

        fig, axes = plt.subplots(nrows=len(list(effective_wf)), sharex=True)
        if not hasattr(axes, '__len__'):
            axes = [axes]

        size = fig.get_size_inches()
        fig.set_size_inches(size[0], size[1]*len(axes)/2)

        labels = []
        ylims = []

        for j, (part, src_snk) in enumerate(sorted(effective_wf)):
            if show_all == False and show_fit and self.fit_args.get((part, src_snk))['overlap'] not in ['ZZ']:
                y_low  = pm(self.get_fits(particles=particles, src_snk=src_snk, **kwargs).p[self.get_p_keys()[(part, src_snk)]['wf']][0], -10)
                y_high = pm(self.get_fits(particles=particles, src_snk=src_snk, **kwargs).p[self.get_p_keys()[(part, src_snk)]['wf']][0], +10)
                ylims.append((y_low, y_high))
            else:
                ylims.append((None, None))

            if self.fit_args.get((part, src_snk))['overlap'] == 'ZZ':
                label = r'$Z^{\rm eff}_{\rm src} Z^{\rm eff}_{\rm snk}$'
            else:
                label = r'$A^{\rm eff}$'
            labels.append(label)
            
        axes = self._plot_quantity(quantity=effective_wf, ax=axes,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel=labels, ylim=ylims, show_legend=show_legend)

        for ax in axes:
            ax.ticklabel_format(axis='y', scilimits=(0,0))
        
        if show_fit:
            fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
            t = np.linspace(t_plot_min, t_plot_max)
            effective_wf_fit = self.fcn_effective_wf(t, particles=particles, src_snk=src_snk, **kwargs)

            for j, (part, src_snk) in enumerate(sorted(effective_wf)):
                if (part, src_snk) in effective_wf_fit:
                    # all args are the same unless all are None

                    color = default_cmap((j+1)/(len(effective_wf)+1))

                    axes[j].plot(t, pm(effective_wf_fit[(part, src_snk)], 0), '--', color=color)
                    axes[j].plot(t, pm(effective_wf_fit[(part, src_snk)], 1), 
                                t, pm(effective_wf_fit[(part, src_snk)], -1), color=color)
                    axes[j].fill_between(t, pm(effective_wf_fit[(part, src_snk)], -1), pm(effective_wf_fit[(part, src_snk)], 1),
                                    facecolor=color, alpha = 0.10, rasterized=True)

                    if fit_args.get((part, src_snk)) is not None:
                        axes[j].axvline(fit_args.get((part, src_snk))['t_start']-0.5, linestyle='--', alpha=0.8, color=color)
                        axes[j].axvline(fit_args.get((part, src_snk))['t_end']-0.5, linestyle='--', alpha=0.8, color=color)
    
        plt.close()
        return fig


    def plot_effective_z(self, show_fit=True, show_all=False, t_plot_min=None, t_plot_max=None, 
            particles=None, src_snk=None, **kwargs):

        tuple_to_str = lambda t : str(t) if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'
        
        if self.get_fits(**kwargs) is None:
            show_fit = False

        if t_plot_min is None:
            t_plot_min, _ = self._set_default_plot_lims(show_fit=show_fit)
        if t_plot_max is None:
            _, t_plot_max = self._set_default_plot_lims(show_fit=show_fit)  

        if t_plot_min is None:
            t_plot_min, _ = self._set_default_plot_lims(show_fit=show_fit)
        if t_plot_max is None:
            _, t_plot_max = self._set_default_plot_lims(show_fit=show_fit)

        if particles is None or particles == 'all':
            particles = self._get_particles()
        elif isinstance(particles, str):
            particles = [particles]

        if src_snk is None:
            part_src_snk_list = [(p,ss) for p, ss in self.correlators if p in particles]
        else:
            part_src_snk_list = [(p,ss) for p, ss in self.correlators 
                if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)]

        fig, axes = plt.subplots(nrows=len(list(part_src_snk_list)), sharex=True)
        if not hasattr(axes, '__len__'):
            axes = [axes]

        size = fig.get_size_inches()
        fig.set_size_inches(size[0], size[1]*len(axes)/2)
            
        effective_z = {}
        labels = {}

        for part, src_snk in sorted(part_src_snk_list):
            # logic only applies if src_snk is a tuple
            if not isinstance(src_snk, str) and hasattr(src_snk, '__len__'):
                for smr in src_snk:
                    if not all([self.effective_z[part, smr][j] is None for j in range(len(self.effective_z[part, smr]))]):
                        effective_z[part, smr] = self.effective_z[part, smr]
                        labels[part, smr] = r'$Z^{\rm eff}$ [%s]'%tuple_to_str((part, smr))
                    else:
                        effective_z[part, src_snk] = np.sqrt(self.effective_wf[part, src_snk])
                        labels[part, src_snk] = r'$\sqrt{A^{\rm eff}}$ [%s]'%tuple_to_str((part, src_snk))
            else:
                effective_z[part, src_snk] = np.sqrt(self.effective_wf[part, src_snk])
                labels[part, src_snk] = r'$\sqrt{A^{\rm eff}}$ [%s]'%tuple_to_str((part, src_snk))
            
        labels = list(labels.values())
            
        axes = self._plot_quantity(quantity=effective_z, ax=axes,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel=labels, ylim=[], show_legend=False)

        for ax in axes:
            ax.ticklabel_format(axis='y', scilimits=(0,0))
        
        if show_fit:
            fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
            t = np.linspace(t_plot_min, t_plot_max)
            effective_z_fit = self.fcn_effective_z(t, particles=particles, src_snk=src_snk, **kwargs)

            for j, (part, src_snk) in enumerate(sorted(effective_z)):
                if (part, src_snk) in effective_z_fit:
                    # all args are the same unless all are None

                    color = default_cmap((j+1)/(len(effective_z)+1))

                    axes[j].plot(t, pm(effective_z_fit[(part, src_snk)], 0), '--', color=color)
                    axes[j].plot(t, pm(effective_z_fit[(part, src_snk)], 1), 
                                t, pm(effective_z_fit[(part, src_snk)], -1), color=color)
                    axes[j].fill_between(t, pm(effective_z_fit[(part, src_snk)], -1), pm(effective_z_fit[(part, src_snk)], 1),
                                    facecolor=color, alpha = 0.10, rasterized=True)

                    if fit_args.get((part, src_snk)) is not None:
                        axes[j].axvline(fit_args.get((part, src_snk))['t_start']-0.5, linestyle='--', alpha=0.8, color=color)
                        axes[j].axvline(fit_args.get((part, src_snk))['t_end']-0.5, linestyle='--', alpha=0.8, color=color)
    
        plt.close()
        return fig


    def plot_stability(self, major_ticks=None, param='E0', show_all=False, show_avg=True, show_best=None, 
            show_legend=None, ylim=None, minor_ticks=None, debug=False, particles=None, src_snk=None, extra_bands=None, **kwargs):

        if 'random_models' in kwargs:
            raise RuntimeError("arg 'random_models' incompatible with plot_stability")

        # useful definitions
        # 1. only get 1st item in list
        to_scalar = lambda l : l[0] if hasattr(l, '__len__')  else l
        # 2. nicely format tuples as strings
        
        if src_snk is None:
            src_snk = 'default'
        else:
            src_snk = fmt_tuple_as_str(src_snk)
        
        if particles is None:
            particles = self._get_particles()
        if isinstance(particles, str):
            particles = [particles]

        particles = [p for p in particles
            if (p, fmt_tuple_as_str(src_snk)) in self.fit_args]   
        
        if particles == []:
            raise ValueError('Invalid combination of particles/src_snk: '
                +str(particles)+' / '+str(src_snk))
 
    
        # get defaults: t_start, t_end, n_states
        fargs_temp  = self.get_fit_args(particles, src_snk)

        defaults = {
            't_start' : np.min([fargs_temp.get((p, ss))['t_start'] for p, ss in fargs_temp 
                                if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)]),
            't_end' : np.max([fargs_temp.get((p, ss))['t_end'] for p, ss in fargs_temp 
                                if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)]),
            'n_states' : np.max([fargs_temp.get((p, ss))['n_states'] for p, ss in fargs_temp 
                                if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)])
        }

        t_start   = kwargs.get('t_start')
        t_end     = kwargs.get('t_end') 
        n_states  = kwargs.get('n_states')
        for key in ['particles', 't_start', 't_end', 'n_states']:
            if key in kwargs:
                del(kwargs[key])

        if major_ticks is None:
            major_ticks = 't_start'

        if isinstance(minor_ticks, str):
            minor_ticks = [minor_ticks]

        elif minor_ticks is None:
            minor_ticks = {}
            minor_ticks['t_start'] = r'$t_{\rm start}=$'
            minor_ticks['n_states'] = r'$N=$'
            minor_ticks['t_end'] = r'$t_{\rm end}=$'
            minor_ticks['energy_gaps'] = r'$dE \propto$'
            minor_ticks['svdcut'] = 'SVD cut $=$'
            minor_ticks['prior_En'] = 'prior $E_n$: '
        
        ic_per_tick = 'logGBF'
        # disable avg for svdcut comparison
        if ('svdcut' in kwargs
            and kwargs['svdcut'] is not None
            and hasattr(kwargs['svdcut'], '__len__')
            and len(kwargs['svdcut']) > 1):
            
            show_avg = False
            ic_per_tick = None

        if major_ticks is None:
            major_ticks = 't_start' 

        if major_ticks  == 't_start':
            xlabel = r'$t_{\rm min}$'
        elif major_ticks == 't_end':
            xlabel = r'$t_{\rm max}$'
        else:
            xlabel = None

        if t_start is None:
            if major_ticks == 't_start':
                if self.auto_fit_args is not None and self.auto_fit_args['enabled']:
                    t_plot_min = self.auto_fit_args['t_start']
                    t_plot_max = self.auto_fit_args['t_end']
                else:
                    t_plot_min = int((4 + defaults['t_start'])/3)
                    if t_end is None:
                        t_plot_max = defaults['t_end'] - 1
                    else:
                        t_plot_max = np.min(t_end) - 1

                t_start = range(t_plot_min, t_plot_max)
            else:
                t_start = defaults['t_start']

        if t_end is None:
            if major_ticks == 't_end':
                if self.auto_fit_args is not None and self.auto_fit_args['enabled']:
                    t_plot_min = self.auto_fit_args['t_start']
                    t_plot_max = self.auto_fit_args['t_end']
                else:
                    t_period = np.min([self._get_t_period(part) for part in particles])
                    t_plot_min = np.min(t_start) + 1
                    t_plot_max = int((2 *defaults['t_end'] + t_period)/3)

                t_end = range(t_plot_min, t_plot_max)

        if n_states is None:
            if self.auto_fit_args is not None and self.auto_fit_args['enabled']:
                n_states = range(1, self.auto_fit_args['n_max']+1)
            else:
                n_states = defaults['n_states'] 

        fargs = {}
        fargs.update(kwargs)
        if t_start is not None:
            fargs['t_start'] = t_start
        if t_end is not None:
            fargs['t_end'] = t_end
        if n_states is not None:
            fargs['n_states'] = n_states

        fits = self.get_fits(particles=particles, src_snk=src_snk, **fargs)
        fit_args_list = self.get_fit_args(particles=particles, src_snk=src_snk, **fargs)
        fit_args_emph = self.get_fit_args(particles=particles, src_snk=src_snk)

        #fits, fit_args_list = self._get_fits(particles=particles, t_start=t_start, t_end=t_end, n_states=n_states, energy_gaps=energy_gaps, svdcut=svdcut)
        #fit_args_emph = self.get_fit_args(particles=particles)
            
        p_keys = self.fitters[self.get_fit_args(particles=particles, **defaults)].p_keys
        if param not in [i for sublist in [list(p_keys[(part, ss)]) for part, ss in p_keys if part in particles] for i in sublist]:
            raise ValueError('Param', param, 'not supported.')
        
        # note: param_keys doesn't have a key for (particle, 'default')
        if src_snk == 'default':
            param_keys = np.unique([p_keys[(p, ss)][param] for p, ss in p_keys if p in particles])
        else:
            param_keys = np.unique([p_keys[(p, ss)][param] for p, ss in p_keys if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)])

        vals_list = [[to_scalar(f.p[key]) for f in fits] for key in param_keys]
        #best_vals = [to_scalar(self.get_fits(particles=particles, t_start=t_start_def, t_end=t_end_def, n_states=n_states_def, energy_gaps=energy_gaps_def, svdcut=svdcut_def).p[key]) for key in param_keys]
        best_vals = [to_scalar(self.get_fits(particles=particles, src_snk=src_snk).p[key]) for key in param_keys]

        if param == 'E0':
            labels=[r'$E_0$ [%s]'%', '.join(key.split('::')[1:]) for key in param_keys]  
        elif param in ['Z_src', 'Z_snk']:
            labels = [r'$Z$ [%s]'%(', '.join(key.split('::')[1:])) for key in param_keys]
        elif param == 'dE':
            labels = [r'$dE$ [%s]'%(', '.join(key.split('::')[1:])) for key in param_keys]
        else:
            labels = [r'%s [%s]'%(param, ', '.join(key.split('::')[1:])) for key in param_keys]

        return self._plot_stability(
            fit_args_list=fit_args_list, vals_list=vals_list, labels=labels, major_ticks=major_ticks, best_vals=best_vals, 
            part_src_snk=(particles[0], src_snk), fit_args_emphasis=fit_args_emph, show_legend=show_legend,
            xlabel=xlabel, show_all=show_all, show_avg=show_avg, show_best=show_best, minor_ticks=minor_ticks, 
            ic_per_tick=ic_per_tick, debug=debug, ylim=ylim, extra_bands=extra_bands)


    def plot_svd_stability(self, param, svdcut_min=1e-12, svdcut_max=0.1, n=11):        
        raise NotImplementedError
        extrapolated_values = []
        fits = []

        svdcuts = np.power(10, np.linspace(np.log10(svdcut_min), np.log10(svdcut_max), n))
        fits = self.get_fits(svdcut=svdcuts)

        extrapolated_values = [
            f.p[param][0] if hasattr(f.p[param], '__len__') 
            else f.p[param]
            for f in fits]

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [5, 1],  'wspace':0.3, 'hspace':0.1})

        axes[0].errorbar(svdcuts, gv.mean(extrapolated_values), xerr=0, yerr=gv.sdev(extrapolated_values), marker='o', mec='white', ls='')
        axes[1].plot(svdcuts, [f.Q for f in fits],  marker='.')

        s = self.fitters[self.fit_args].svdcut
        if s is not None:
            axes[0].axvline(s, ls='--', color='red')
            axes[1].axvline(s, ls='--', color='red')
        
        axes[0].set_ylabel(param)
        axes[1].set_ylabel('$Q$')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_xlabel('svd cut')
        axes[1].set_xscale('log')

        plt.close()
        return fig


    def plot_supereffective_mass(self, tau, t_plot_min=None, t_plot_max=None, ylim=None):
        raise NotImplementedError
        supereffective_mass = {}
        for part_src_snk in list(self.correlators):
            corr = self.correlators[part_src_snk]

            num = np.roll(corr,  0) - (np.roll(corr, -tau)/np.roll(corr, -(tau+1) )) *np.roll(corr, -1)
            den = np.roll(corr, -1) - (np.roll(corr, -tau)/np.roll(corr, -(tau+1) )) *np.roll(corr, -2)
            supereffective_mass[part_src_snk] = np.log(num/den) - np.log(np.roll(corr,  -tau) / np.roll(corr, -(tau+1) ))

        fig = self._plot_quantity(
            quantity=supereffective_mass,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel=r'$dM_{\rm eff}(t)$', ylim=ylim)
        return fig