import matplotlib.pyplot as plt
import numpy as np
import gvar as gv
import matplotlib

import corrfit.base
from corrfit.plot import default_cmap
from corrfit.utils import fmt_tuple_as_str

class FitManager(corrfit.base.FitManager):

    def __init__(self, data, prior=None, fargs_unfmt=None):

        self.data = data
        self.prior = prior
        self._fargs_unfmt = fargs_unfmt
        self._fitters = None


    @property
    def fit_args(self):
        return corrfit.blossier.fit_args.FitArgs(fargs_unfmt=self._fargs_unfmt, data_keys=list(self.data))


    @property
    def fitters(self):
        if self._fitters is None:
            self._fitters = corrfit.blossier.fitters_dict.FittersDict(
                data=self.data, prior=self.prior)
        return self._fitters
    

    def _get_models(self, particles=None, src_snk=None, **kwargs):
        fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
        models = self.fitters[fit_args].models
        output = {model.datatag : model for model in models} 
        output = {k : v for k, v in sorted(output.items())}
        return output
    

    def fcn_linexp(self, t, particles=None, src_snk=None, **kwargs):
        p = self.get_fits(particles=particles, src_snk=src_snk, **kwargs).p
        output = {}
        for datatag, model in sorted(self._get_models(particles=particles, src_snk=src_snk, **kwargs).items()):
            if datatag[0] == 'linexp':
                output[(datatag[1], datatag[2])] = model.fitfcn(p, t)
        return output
    

    def plot_data(self, show_fit=True, t_plot_min=None, t_plot_max=None, show_legend=None, split_axes=False, 
            ylim=None, particles=None, src_snk=None, show_all=None, extra_bands={}, ylabel=None, max_j=None, **kwargs):
        pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
        
        # offset from excluding some eigenvalues
        x_offset = 4

        if not show_fit or show_fit is None or self.get_fits(particles=particles, src_snk=src_snk, **kwargs) is None:
            show_fit = False

        fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
        
        if t_plot_min is None:
            t_plot_min = 0
        else:
            t_plot_min = t_plot_min - 4
        
        if t_plot_max is None:
            t_plot_max = np.nanmin([len(self.data[p_ss]) for p_ss in self.data])
        else:
            t_plot_max = t_plot_max - 4

        if particles is None or particles == 'all':
            particles = [p for p, _ in self.data]
        elif isinstance(particles, str):
            particles = [particles]

        if src_snk is None:
            data = {p_ss : self.data[p_ss] for p_ss in self.data if p_ss[0] in particles}
        else:
            data = {(p, ss) : self.data[(p, ss)] for p, ss in self.data 
                if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)}
            
        if max_j is None:
            max_j = len(list(data))

        if split_axes:
            fig, axes = plt.subplots(nrows=max_j, sharex=True)
            size = fig.get_size_inches()
            fig.set_size_inches(size[0], size[1]*len(axes)/2)

            if show_legend is None:
                show_legend = False

            labels = []
            if fit_args == {}:
                for p_ss in data:
                    if show_legend:
                        labels.append(None)
                    else:
                        labels.append(fmt_tuple_as_str(p_ss))
            else:
                for p_ss in data:
                    if show_legend:
                        labels.append(fit_args.get(p_ss)['xi'])
                    else:
                        labels.append(fit_args.get(p_ss)['xi']+' [%s]'%(fmt_tuple_as_str(p_ss)))#tuple_to_str((p, ss)))

        else:
            fig, axes = plt.subplots()

            if show_legend is None:
                show_legend = True
            
            if fit_args is None:
                labels = None
            else:
                labels = ' // '.join(np.unique([fit_args.get(p_ss)['xi'] for p_ss in fit_args]))

        if ylabel is not None:
            labels = ylabel

        #if not hasattr(axes, '__len__'):
        #    axes = [axes]

        if show_all:
            ylim = None
        elif (show_fit and ylim is None) or (ylim == 'prior'):
            xi_key = self.fit_args[list(self.fit_args)[0]]['xi']
            if src_snk is None:
                ylim = [(pm(self.prior[k], -1), pm(self.prior[k], 1)) for k in self.prior 
                            if k.startswith(xi_key)]
            else:
                ylim = [(pm(self.prior[k], -1), pm(self.prior[k], 1)) for k in self.prior 
                            if k.startswith(xi_key) and k.endswith(fmt_tuple_as_str(src_snk))]

        if extra_bands is not None and not hasattr(axes, '__len__'):
            for j, (key, val) in enumerate(extra_bands.items()):
                cmap = matplotlib.colormaps['spring']
                color = cmap((j+1)/(len(extra_bands)+1))
                axes.axhspan(pm(val, -1), pm(val, 1), label=key, alpha=0.5, color=color)

        axes = self._plot_quantity(
            quantity=data, 
            ax=axes if split_axes else axes,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel=labels, show_legend=show_legend, ylim=ylim,
            x_offset=x_offset, max_j=max_j)
        
        if show_fit:
            t = np.linspace(t_plot_min-0.5, t_plot_max+0.5)
            data_fit = self.fcn_linexp(t+x_offset, particles, src_snk, **kwargs)
            if src_snk is None:
                data_fit = {p_ss : data_fit[p_ss] for p_ss in data_fit if p_ss[0] in particles}
            else:
                data_fit = {(p, ss) : data_fit[(p, ss)] for p, ss in data_fit 
                    if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)}

            for j, (p, ss) in enumerate(data):
                if (p, ss) in data_fit and j < max_j:
                    if split_axes:
                        ax = axes[j]
                    else:
                        ax = axes
                    color = default_cmap((j+1)/(len(data)+1))

                    ax.plot(t+x_offset, pm(data_fit[(p, ss)], 0), '--', color=color)
                    ax.plot(t+x_offset, pm(data_fit[(p, ss)], 1), 
                                t+x_offset, pm(data_fit[(p, ss)], -1), color=color)
                    ax.fill_between(t+x_offset, pm(data_fit[(p, ss)], -1), pm(data_fit[(p, ss)], 1),
                                    facecolor=color, alpha = 0.10, rasterized=True)

                    if fit_args.get((p, ss)) is not None:
                        if split_axes:
                            linestyle = '--'
                        else:
                            linestyle = (j, (1, len(data_fit)))
                        ax.axvline(fit_args.get((p, ss)) ['t_start']-0.5, linestyle=linestyle, alpha=0.8, color=color)
                        ax.axvline(fit_args.get((p, ss)) ['t_end']-0.5, linestyle=linestyle, alpha=0.8, color=color)

        plt.close()
        return fig
    

    def plot_stability(self, major_ticks=None, show_all=False, show_avg=True, show_best=None, minor_ticks=None, 
            ylim=None, particles=None, src_snk=None, debug=False, show_legend=True, ncols=1, extra_bands=None, ylabels=None, **kwargs):

        if 'random_models' in kwargs:
            raise RuntimeError("arg 'random_models' incompatible with plot_stability")

        if src_snk is not None:
            src_snk = fmt_tuple_as_str(src_snk)

        if particles is None or particles == 'all':
            particles = [p for p, _ in self.data]
        elif isinstance(particles, str):
            particles = [particles]

        if src_snk is None:
            particles = np.unique([p for p in particles
                if (p, 'default') in self.fit_args])
        else:
            particles = np.unique([p for p in particles
                if (p, fmt_tuple_as_str(src_snk)) in self.fit_args] )
        
        # get defaults: t_start, t_end, n_states
        fargs_temp  = self.get_fit_args(particles, src_snk)

        defaults = {
            't_start' : [], 
            't_end' : [], 
            'n_states' : [], 
        }
        for p, ss in fargs_temp:
            if p in particles:
                if (src_snk is None) or (fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)):
                    defaults['t_start'].append(fargs_temp.get((p, ss))['t_start'])
                    defaults['t_end'].append(fargs_temp.get((p, ss))['t_end'])
                    defaults['n_states'].append(fargs_temp.get((p, ss))['n_states'])
        defaults['t_start'] = np.min([defaults['t_start']])
        defaults['t_end'] = np.min([defaults['t_end']])
        defaults['n_states'] = np.min([defaults['n_states']])

        if src_snk is None:
            src_snk = 'default'

        t_start   = kwargs.get('t_start')
        t_end     = kwargs.get('t_end') 
        n_states  = kwargs.get('n_states') or defaults['n_states']
        for key in ['t_start', 't_end', 'n_states']:
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
                t_plot_min = int((8 + defaults['t_start'])/3)
                if t_end is None:
                    t_plot_max = defaults['t_end'] - 1
                else:
                    t_plot_max = np.min(t_end) - 1

                t_start = range(t_plot_min, t_plot_max)
            else:
                t_start = defaults['t_start']

        if t_end is None:
            if major_ticks == 't_end':
                t_period = len(self.data[list(self.data)[0]])
                t_plot_min = np.min(t_start) + 1
                t_plot_max = int((2 *defaults['t_end'] + t_period)/3)

                t_end = range(t_plot_min, t_plot_max)

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
        fit_args_emph = self.get_fit_args(particles=particles, src_snk=src_snk, n_states=n_states)

        p_keys = self.fitters[self.get_fit_args(particles=particles, **defaults)].p_keys
        # note: param_keys doesn't have a key for (particle, 'default')
        if src_snk == 'default':
            param_keys = np.unique([p_keys[(p, ss)]['xi'] for p, ss in p_keys if p in particles])
        else:
            param_keys = np.unique([p_keys[(p, ss)]['xi'] for p, ss in p_keys if p in particles and fmt_tuple_as_str(ss) == fmt_tuple_as_str(src_snk)])

        vals_list = [[f.p[key] if f is not None else None for f in fits] for key in param_keys]
        best_vals = [self.get_fits(particles=particles, src_snk=src_snk).p[key]
            for key in param_keys]
        
        if ylabels is None:
            labels = [r'$xi$ [%s]'%', '.join(key.split('::')[1:]) for key in param_keys]
        elif isinstance(ylabels, str):
            labels = [ylabels]
        else:
            labels = ylabels

        #return len(fit_args_list), len(vals_list[0]), len(best_vals)

        ic_per_tick = 'logGBF'
        if np.any([kwargs.get(k) is not None and not isinstance(kwargs.get(k), str) and hasattr(kwargs.get(k), '__len__') and len(kwargs.get(k)) > 1 
                for k in ['svdcut', 'uncorrelated']]):
            ic_per_tick=None
            show_avg=False
        
        if major_ticks != 't_start' and hasattr(t_start, '__len__') and len(t_start) > 1:
            ic_per_tick=None
        if major_ticks != 't_end' and hasattr(t_end, '__len__') and len(t_end) > 1:
            ic_per_tick=None

        return self._plot_stability(
            fit_args_list=fit_args_list, vals_list=vals_list, labels=labels, major_ticks=major_ticks, best_vals=best_vals, 
            part_src_snk=(particles[0], src_snk), fit_args_emphasis=fit_args_emph, 
            xlabel=xlabel, show_all=show_all, show_avg=show_avg, show_best=show_best, minor_ticks=minor_ticks, 
            ic_per_tick=ic_per_tick, ylim=ylim, debug=debug, show_legend=show_legend, ncols=ncols, extra_bands=extra_bands)
