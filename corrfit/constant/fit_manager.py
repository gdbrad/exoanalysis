import numpy as np
import copy
import tqdm
import itertools
import matplotlib
import matplotlib.pyplot as plt
import gvar as gv

import corrfit.base
import corrfit.constant.fit_args
import corrfit.constant.fitters_dict

class FitManager(corrfit.base.FitManager):

    def __init__(self, data, prior=None, fit_args=None, svdcut=None):

        self.data = data
        self.prior = prior
        self.svdcut = svdcut
        if fit_args is not None:
            self.fit_args = corrfit.constant.fit_args.FitArgs(fit_args)
        else:
            self.fit_args = None

        self._fitters = None


    @property
    def fitters(self):
        if self._fitters is None:
            self._fitters = corrfit.constant.fitters_dict.FittersDict(self.data, self.prior, svdcut=self.svdcut)
        return self._fitters
        

    def __str__(self):
        fit_args = self.fit_args
        output = 'Fit args:\n'
        max_len = np.max([len(particle) for particle in fit_args])
        for particle in fit_args:
            output += (particle+':').rjust(max_len+3)+'   '
            if self.fit_args[particle]['perform_fit']:
                output += 't = [%s, %s)\n'%(fit_args[particle]['t_start'], fit_args[particle]['t_end'])
            else:
                output += 'No fit performed\n'
        output += '\n'
        output += str(self.get_fit())
        return output
    

    def _get_fit_args(self, particles=None, t_start=None, t_end=None):
        if self.fit_args is None:
            raise NameError('fit_args unspecified when instantiating FitManager object')
        
        if particles is None or particles == 'all':
            particles = [p for p in self.fit_args]
        elif isinstance(particles, str):
            particles = [particles]

        if t_start is None and t_end is None:
            fit_args = {}
            for part in particles:
                fit_args[part] = copy.deepcopy(self.fit_args[part])

        else:
            if t_start is None:
                t_start = np.nanmin([self.fit_args[p]['t_start'] for p in particles])
            if t_end is None:
                t_end = np.nanmax([self.fit_args[p]['t_end'] for p in particles])

            fit_args = {}
            for part in particles:
                fit_args[part] = {}
                fit_args[part]['t_start'] = t_start
                fit_args[part]['t_end'] = t_end

        return corrfit.constant.fit_args.FitArgs(fit_args=fit_args)


    def _get_fits(self, particles=None, t_start=None, t_end=None):
        if particles is None or particles == 'all':
            particles = [p for p in self.fit_args]
        elif isinstance(particles, str):
            particles = [particles]

        if t_start is None:
            t_start = np.nanmin([self.fit_args[p]['t_start'] for p in particles])
        if not hasattr(t_start, '__len__'):
            t_start = [t_start]

        if t_end is None:
            t_end = np.nanmax([self.fit_args[p]['t_end'] for p in particles])
        if not hasattr(t_end, '__len__'):
            t_end = [t_end]

        fits = []
        fit_args = []
        for t1, t2 in tqdm.tqdm(itertools.product(t_start, t_end), 
                                        desc='Collecting fits', total=len(t_start) *len(t_end)):
            if t1 < t2:
                fit_args.append(self._get_fit_args(particles=particles, t_start=t1, t_end=t2))
                fits.append(self.get_fit(particles=particles, t_start=t1, t_end=t2))
                
        return fits, fit_args


    def get_fit(self, particles=None, t_start=None, t_end=None):
        fit_args = self._get_fit_args(particles=particles, t_start=t_start, t_end=t_end)
        return self.fitters[fit_args].fit
    

    def plot_data(self, particles=None, t_start=None, t_end=None, t_plot_min=None, t_plot_max=None, show_fit=True, ylim=None):
        tuple_to_str = lambda t : t if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'
        pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
        
        if self.fit_args is None:
            show_fit = False
        
        if show_fit:
            fit_args = self._get_fit_args(particles=particles, t_start=t_start, t_end=t_end)

        if particles is None:
            particles = [part for part, _ in self.data]
        elif isinstance(particles, str):
            particles = [particles]

        data = {p_ss  : self.data[p_ss] for p_ss in self.data if p_ss[0] in particles}

        if t_plot_min is None:
            t_plot_min = 1
        if t_plot_max is None:
            t_plot_max = np.nanmin([len(data[k]) for k in data])

        fig, axes = plt.subplots(nrows=len(list(data)), sharex=True)
        if len(list(data)) == 1:
            axes = [axes]

        size = fig.get_size_inches()
        fig.set_size_inches(size[0], size[1]*len(axes)/2)

        labels = []
        #ylims = []

        for j, (part, src_snk) in enumerate(sorted(data)):
            labels.append('%s, %s'%(part, (tuple_to_str(src_snk))))
            
        axes = self._plot_quantity(quantity=data, ax=axes,
            t_plot_min=t_plot_min, t_plot_max=t_plot_max, 
            ylabel=labels, ylim=ylim, show_legend=False)

        for ax in axes:
            ax.ticklabel_format(axis='y', scilimits=(0,0))

        if show_fit:
            t = np.linspace(t_plot_min, t_plot_max)
            p_keys = self.fitters[fit_args].p_keys
            fit = self.fitters[fit_args].fit
            fitted_data = {(part, src_snk) : np.repeat(fit.p[p_keys[(part, src_snk)]['c']], len(t)) 
                for (part, src_snk) in p_keys}

            colors1 = matplotlib.cm.get_cmap('cool_r')(np.linspace(0., 1, 128))
            colors2 = matplotlib.cm.get_cmap('winter_r')(np.linspace(0, 1, 128))
            colors = np.vstack((colors1, colors2))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
            
            for j, (part, src_snk) in enumerate(sorted(data)):
                if (part, src_snk) in fitted_data:
                    color = cmap((j+1)/(len(data)+1))

                    axes[j].plot(t, pm(fitted_data[(part, src_snk)], 0), '--', color=color)
                    axes[j].plot(t, pm(fitted_data[(part, src_snk)], 1), 
                        t, pm(fitted_data[(part, src_snk)], -1), color=color)
                    axes[j].fill_between(t, pm(fitted_data[(part, src_snk)], -1), pm(fitted_data[(part, src_snk)], 1),
                        facecolor=color, alpha = 0.10, rasterized=True)
                    
                    t1 = fit_args[part]['t_start']
                    t2 = fit_args[part]['t_end']
                    axes[j].axvline(t1-0.5, linestyle='--', alpha=0.8, color=color)
                    axes[j].axvline(t2-0.5, linestyle='--', alpha=0.8, color=color)

        plt.close()
        return fig
    

    def plot_stabiltiy(self, major_ticks=None, particles=None, t_start=None, t_end=None, 
                    part_src_snk=None, show_all=False, show_avg=True, show_best=None, minor_ticks=None, debug=False):
        
        if particles is None:
            particles = [p for p in self.fit_args]

        if isinstance(particles, str):
            particles = [particles]
        if isinstance(minor_ticks, str):
            minor_ticks = [minor_ticks]
        elif minor_ticks is None:
            minor_ticks = {}
            minor_ticks['t_start'] = r'$t_{\rm start}=$'
            minor_ticks['t_end'] = r'$t_{\rm end}=$'

        if t_start is None:
            t_start_def = np.nanmin([self.fit_args[p]['t_start'] for p in self.fit_args if p in particles])
        else:
            t_start_def = np.nanmin(t_start)

        if t_end is None:
            t_end_def = np.nanmax([self.fit_args[p]['t_end'] for p in self.fit_args if p in particles])
        else:
            t_end_def = np.nanmax(t_end)

        t_period = np.nanmin([len(self.data[p_ss]) for p_ss in self.data if p_ss[0] in particles])

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
                t_plot_min = int((4 + t_start_def)/3)
                t_plot_max = t_end_def - 1
                
                t_start = range(t_plot_min, t_plot_max)
            else:
                t_start = [t_start_def]

        if t_end is None:
            if major_ticks == 't_end':
                t_plot_min = t_start_def + 1
                t_plot_max = int((2 *t_end_def + t_period)/3)

                t_end = range(t_plot_min, t_plot_max)
            else:
                t_end = [t_end_def]

        fits, fit_args_list = self._get_fits(particles=particles, t_start=t_start, t_end=t_end)
        fit_args_emph = self._get_fit_args(particles=particles)
        print(fit_args_emph)

        p_keys = self.fitters[self._get_fit_args(particles=particles, t_start=t_start_def, t_end=t_end_def)].p_keys
        if part_src_snk is None:
            param_keys = np.unique([p_keys[(part, src_snk)]['c'] for part, src_snk in p_keys if part in particles])
        else:
            param_keys = np.unique([p_keys[(part, src_snk)]['c'] for part, src_snk in p_keys if (part, src_snk) == part_src_snk])

        vals_list = [[f.p[key] for f in fits] for key in param_keys]
        best_vals = [self.get_fit(particles=particles, t_start=t_start_def, t_end=t_end_def).p[key] for key in param_keys]
        labels=[r'$c$ [%s]'%', '.join(key.split('::')[1:]) for key in param_keys]  

        return self._plot_stability(fit_args_list, vals_list, labels, major_ticks, best_vals, fit_args_emphasis=fit_args_emph, particles=particles, xlabel=xlabel, show_all=show_all, show_avg=show_avg, show_best=show_best, minor_ticks=minor_ticks, debug=debug)