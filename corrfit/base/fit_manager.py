import lsqfitics
import matplotlib
import matplotlib.pyplot as plt
import functools
import numpy as np
import gvar as gv
import re
import abc
import tqdm
import fnmatch

from ..plot import default_cmap, get_closest_factorization


class FitManager(abc.ABC):

    def __init__(self):
        pass


    def __str__(self):
        if self.get_fit_args() == {}:
            return 'No fit performed'

        output = '~'*15 +' FIT SETTINGS ' + '~'*15 +'\n'
        output += str(self.fit_args)
        output += '\n'
        output += '~'*15 +' FIT RESULTS ' + '~'*15 +'\n'
        output += str(self.get_fits())
        return output

    @property
    @abc.abstractmethod
    def fitters(self):
        return None
    
    
    def _match_parameters_fits(self, params, **kwargs):
        if not hasattr(params, '__len__') or isinstance(params, str):
            params = [params]

        if isinstance(params, str):
            params = [params]

        fits = self.get_fits(**kwargs)
        if not hasattr(fits, '__len__'):
            fits = [fits]
        
        # allows wildcard in params, eg 'E::*'
        matched_params = []
        for p in params:
            for f in fits:
                matched_params.extend(fnmatch.filter(list(f.p), p))
        params = np.unique(matched_params)

        return params, fits


    def average_parameters(self, params, ic='BAIC', **kwargs):
        params, fits = self._match_parameters_fits(params, **kwargs)
        
        values = { p :
            [f.p[p] if p in f.p else np.nan for f in fits]
            for p in params
        }
        output = {p : self._calculate_average(values[p], fits=fits, ic=ic) for p in params}
        return output


    def _calculate_weights(self, fits, ic='BAIC'):
        weights = lsqfitics.calculate_weights(fits, ic=ic)
        return weights


    def _calculate_average(self, values, weights=None, fits=None, ic='BAIC'):
        if weights is None:
            if fits is None:
                raise ValueError("Must specify either values or weights")
            weights = self._calculate_weights(fits, ic=ic)

        avg = lsqfitics.calculate_average(values, weights)
        return avg
    
    
    def _sort_fit_args(self, fit_args_list, ic, cdf=0.99, cutoff=0.00, n=None):
        sorted_indices = lsqfitics.argsort(
            [self.fitters[farg].fit for farg in fit_args_list], 
            ic=ic, cdf=cdf, cutoff=cutoff, n=n)
        return sorted_indices
    

    def get_fit_args(self, particles=None, src_snk=None, **kwargs):
        debug = False
        if 'debug' in kwargs:
            debug = kwargs['debug']
            del(kwargs['debug'])

        output = self.fit_args.cartesian_product(particles=particles, src_snk=src_snk, **kwargs)
        if debug:
            print('Debugging....')
            for j, o in enumerate(output):
                print(str(j+1)+' / '+str(len(output)))
                print(o)

        if len(output) == 1:
            return output[0]
        else:
            return output


    def get_fits(self, particles=None, src_snk=None,  **kwargs):
        fit_args = self.get_fit_args(particles=particles, src_snk=src_snk, **kwargs)
        if isinstance(fit_args, list):
            return [self.fitters[fargs].fit 
                for fargs in tqdm.tqdm(fit_args, desc="Collecting fits: ")] 
        else:
            return self.fitters[fit_args].fit
        

    def plot_CDF(self, params, grid_shape=None, ic='BAIC', xlabel=None, **kwargs):
        if isinstance(xlabel, str):
            xlabel = [xlabel]
            
        params, fits = self._match_parameters_fits(params, **kwargs)
        weights = self._calculate_weights(fits, ic=ic)
        
        values = { p :
            [f.p[p] if p in f.p else np.nan for f in fits]
            for p in params
        }

        if grid_shape is None:
            grid_shape = get_closest_factorization(len(params))

        fig, axes = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], gridspec_kw={'wspace':0.2, 'hspace':0.2}, squeeze=0)
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] *grid_shape[1], size[1] *grid_shape[0])

        for j in range(grid_shape[0] * grid_shape[1] - len(params)):
            fig.delaxes(axes.flatten()[-(j+1)])


        for j, (p, ax) in enumerate(zip(params, axes.ravel())):
            ax = lsqfitics.plot_CDF(values=values[p], weights=weights, ax=ax)
            if xlabel is None:
                ax.set_xlabel(p)
            else:
                ax.set_xlabel(xlabel[j])

        plt.close()
        return fig
    

    def plot_histogram(self, params, grid_shape=None, ic='BAIC', xlabel=None, **kwargs):
        if isinstance(xlabel, str):
            xlabel = [xlabel]
    
        params, fits = self._match_parameters_fits(params, **kwargs)
        weights = self._calculate_weights(fits, ic=ic)
        
        values = { p :
            [f.p[p] if p in f.p else np.nan for f in fits]
            for p in params
        }

        if grid_shape is None:
            grid_shape = get_closest_factorization(len(params))

        fig, axes = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], gridspec_kw={'wspace':0.2, 'hspace':0.2}, squeeze=0)
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] *grid_shape[1], size[1] *grid_shape[0])

        for j in range(grid_shape[0] * grid_shape[1] - len(params)):
            fig.delaxes(axes.flatten()[-(j+1)])

        for j, (p, ax) in enumerate(zip(params, axes.ravel())):
            ax = lsqfitics.plot_histogram(values=values[p], weights=weights, ax=ax)
            if xlabel is None:
                ax.set_xlabel(p)
            else:
                ax.set_xlabel(xlabel[j])
                

        plt.close()
        return fig
    

    def _plot_quantity(self, quantity, ax=None,
            t_plot_min=None, t_plot_max=None, 
            ylabel=None, ylim=None, show_legend=None, as_fig=False, 
            x_offset=0, max_j=None):
        
        if show_legend is None and len(quantity) > 4:
            show_legend = False
        elif show_legend is None:
            show_legend = True

        if max_j == None:
            max_j = len(list(quantity))

        # returns string or better formats tuples
        tuple_to_str = lambda t : t if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'

        if ax is None:
            fig, ax = plt.subplots()

        if t_plot_min == None: t_plot_min = 1
        if t_plot_max == None: t_plot_max = len(quantity[list(quantity)[0]]) - 1

        for j, part_src_snk in enumerate(sorted(quantity)):
            if j < max_j:
                if hasattr(ax, '__len__'):
                    ax_temp = ax[j]
                    dx = 0
                else:
                    ax_temp = ax
                    dx = (np.array(range(-len(quantity)+1, len(quantity)+1))[::2] / (4*len(quantity)))[j]
                x = np.arange(t_plot_min, t_plot_max)

                y = gv.mean(quantity[part_src_snk])[x]
                y_err = gv.sdev(quantity[part_src_snk])[x]

                #linewidth = 12 / len(quantity)
                color = default_cmap((j+1)/(len(quantity)+1))
                #ax_temp.errorbar(x+dx, y, xerr = 0.0, yerr=y_err, fmt='o', color=color, label=tuple_to_str(src_snk),
                #            alpha=0.6, mec='white', elinewidth=elinewidth)
                ax_temp.errorbar(x+dx+x_offset, y, xerr = 0.0, yerr=y_err,
                    ls='', marker='o', color=color, markeredgecolor='k', 
                    label=tuple_to_str((part_src_snk[0], tuple_to_str(part_src_snk[1]))))

        if hasattr(ax, '__len__'):
            axes = ax
        else:
            axes = [ax]

        for j, ax_temp in enumerate(axes):
            if j < max_j:
                for t in range(t_plot_min+x_offset, t_plot_max+1+x_offset):
                    ax_temp.axvline(t-0.5, alpha=0.1, ls='--')

                # Label dirac/smeared data
                if show_legend:
                    ax_temp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                if ylabel is not None:
                    if isinstance(ylabel, str):
                        ax_temp.set_ylabel(ylabel)
                    else: # must be a list of strings
                        ax_temp.set_ylabel(ylabel[j])

                
                if ylim is not None and len(ylim) > 0:
                    if hasattr(ylim[0], '__len__'):
                        if ylim[j] is not None:
                            ax_temp.set_ylim(ylim[j])
                    else:
                        ax_temp.set_ylim(ylim)

                ax_temp.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                ax_temp.set_xlim(t_plot_min - 0.6 + x_offset, t_plot_max - 0.4 + x_offset)

        axes[-1].set_xlabel('$t/a$')

        if as_fig:
            plt.close()
            return fig

        return ax
    

    def _plot_stability(self, fit_args_list, vals_list, labels, major_ticks, part_src_snk, best_vals=None, fit_args_emphasis=None, 
                    minor_ticks=None, show_all=False, show_avg=False, show_best=None, ic='BAIC', ic_per_tick='logGBF', xlabel=None, 
                    ylim=None, debug=False, ncols=1, show_legend=True, extra_bands=None):
            
            if debug:
                for farg in fit_args_list:
                    print('\n'+'---'*15)
                    print(farg)
                
                if fit_args_emphasis is not None:
                    print('\n'+'---'*15)
                    print('Emphasized fit arg')
                    print('---'*15)
                    print(fit_args_emphasis)

            if show_best is None:
                show_best = bool(fit_args_emphasis in fit_args_list)

            fmt_label = {}
            if isinstance(minor_ticks, dict):
                for key in minor_ticks:
                    fmt_label[key] = minor_ticks[key]
                minor_ticks = list(minor_ticks)

            # works if 'None' in l
            def get_unique(input_list):
                return functools.reduce(lambda l, x: l.append(x) or l if x not in l else l, input_list, [])

            pm = lambda g, k : gv.mean(g) + k* gv.sdev(g)

            # get all unique fit arg keys
            temp = []
            for fargs in fit_args_list:
                temp.extend(list(fargs.full_paths(fargs[part_src_snk])))

            all_fit_arg_keys = get_unique(temp)
            for key in all_fit_arg_keys:
                if key not in fmt_label:
                    fmt_label[key] = key+':'

            # collect unique values for fit_arg keys
            varied_keys = {}
            varied_keys[major_ticks] = [] # first entry is major ticks -- useful for indexing dx
            # first sort ordering per minor ticks
            if minor_ticks is not None:
                for key in minor_ticks:
                    varied_keys[key] = []
            for fargs in fit_args_list:
                for path in all_fit_arg_keys:
                    #print(part, path)
                    val = fargs.get_from_path(path, part_src_snk=part_src_snk)
                    if val is None:
                        pass
                    elif path in varied_keys:
                        varied_keys[path].append(val)
                    else:
                        varied_keys[path] = [val]

            for path in list(varied_keys):
                varied_keys[path] = get_unique(varied_keys[path])
                if len(varied_keys[path]) in [0, 1]:
                    del(varied_keys[path])

            if len(varied_keys) == 1:
                ic_per_tick = None

            if np.isreal(varied_keys[major_ticks]).all(): # check whether number-like
                x_major_ticks = varied_keys[major_ticks]
            else:
                x_major_ticks = np.arange(len(varied_keys[major_ticks]))
            num_divs = np.prod([len(varied_keys[k]) for k in varied_keys if k != major_ticks])
            x_minor_ticks = np.linspace((1 - num_divs)/(2 *1.5 *num_divs), (num_divs - 1)/(2 *1.5 *num_divs), 
                int(np.prod([len(varied_keys[k]) for k in varied_keys if k != major_ticks])))
            
            cmap = matplotlib.colormaps['cool']

            xy = { l : 
                {'x' : [], 'y' : [], 'q' : [], 'chi2r' : [], 'label' : [], 'color' : [],}# 'fargs' : []}
                for l in labels}
            for j, fargs in enumerate(fit_args_list):
                dx_index = int(np.sum([varied_keys[k].index(fargs.get_from_path(k, part_src_snk=part_src_snk)) *int(np.prod([len(varied_keys[l]) for l in list(varied_keys)[j+1:]])) 
                    for j, k in enumerate(varied_keys) if k!= major_ticks]))
                dx = x_minor_ticks[dx_index]

                x_index = varied_keys[major_ticks].index(fargs.get_from_path(major_ticks, part_src_snk=part_src_snk))
                x = (x_major_ticks[x_index] + dx)
                fit = self.fitters[fargs].fit
                for k, l in enumerate(labels):
                    legend_label = ', '.join([fmt_label[k] +' '+ str(fargs.get_from_path(k, part_src_snk=part_src_snk)) for k in varied_keys if k != major_ticks])
                    xy[l]['x'].append(x),
                    xy[l]['y'].append(vals_list[k][j]),
                    xy[l]['q'].append(fit.Q),
                    xy[l]['chi2r'].append(fit.chi2/fit.dof),
                    xy[l]['label'].append(legend_label)
                    xy[l]['color'].append(cmap((dx_index+1)/(len(x_minor_ticks)+1)))

            num_misc_axes = 2 + int(bool(show_avg == True)) + int(bool(ic_per_tick is not None))
            fig, axes = plt.subplots(nrows=len(labels) + num_misc_axes, sharex=True, 
                gridspec_kw={'height_ratios': [5]*len(labels) + [1 *ncols] *num_misc_axes, 'wspace':0.1, 'hspace':0.1})
            axes_params, axes_stats = axes[:len(labels)], axes[len(labels):]
            size = fig.get_size_inches()

            fig.set_size_inches(size[0] *(3 + 3*ncols)/7, size[1]*(3 + 4*np.ceil(len(labels)/ncols))/7)

            for j, l in enumerate(labels):
                axes_params[j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                axes_params[j].set_ylabel(l)
                if ylim is not None:
                    axes_params[j].set_ylim(ylim)

                if best_vals is not None and show_best:
                    dx_index = int(np.sum([np.argwhere(varied_keys[k] == fit_args_emphasis.get_from_path(k, part_src_snk=part_src_snk)) *int(np.prod([len(varied_keys[l]) for l in list(varied_keys)[j+1:]])) 
                        for j, k in enumerate(varied_keys) if k!= major_ticks]))
                    
                    color = cmap((dx_index+1)/(len(x_minor_ticks)+1))
                    axes_params[j].axhspan(pm(best_vals[j], -1), pm(best_vals[j], +1), alpha=0.5, color=color, label='Best')
                    if not show_all and ylim is None:
                        axes_params[j].set_ylim(pm(best_vals[j], -4), pm(best_vals[j], +4))

                if show_avg:
                    ord = np.argsort(xy[l]['x']) # fix ordering of x & y in axis
                    fits_list = np.array([self.fitters[fargs].fit for fargs in fit_args_list])[ord]
                    weights = lsqfitics.calculate_weights(fits_list, ic=ic)
                    avg = lsqfitics.calculate_average(np.array(xy[l]['y'])[ord], weights=weights)
                    axes_params[j].axhspan(pm(avg, -1), pm(avg, 1), alpha=0.5, color='springgreen', label=ic)
                    if j == 0:
                        axes_stats[-1] = lsqfitics.plot_weights(fits_list, ax=axes_stats[-1], x=np.sort(xy[l]['x']), 
                            ics=[ic], show_legend=show_legend)
                    if not show_all and ylim is None and best_vals is not None:
                        axes_params[j].set_ylim(pm(best_vals[j], -4), pm(best_vals[j], +4))

                if extra_bands is not None:
                    for i, (key, val) in enumerate(extra_bands.items()):
                        cmap = matplotlib.colormaps['spring']
                        color = cmap((i+1)/(len(extra_bands)+1))
                        axes_params[j].axhspan(pm(val, -1), pm(val, 1), label=key, alpha=0.5, color=color)

                if ic_per_tick is not None and j == 0:
                    for xm, key_val in zip(x_major_ticks, varied_keys[major_ticks]):
                        xvals = np.array([xy[l]['x'][k] for k in range(len(xy[l]['x']))
                            if xm - 0.5 < xy[l]['x'][k] < xm + 0.5])
                        ord = np.argsort(xvals)
                        fits_list = np.array([self.fitters[fargs].fit for fargs in fit_args_list
                                            if fargs.get_from_path(major_ticks, part_src_snk=part_src_snk) == key_val])[ord]
                        weights = lsqfitics.calculate_weights(fits_list, ic=ic_per_tick)
                        color = xy[l]['color'][np.argmax(weights)]
                        if xm == x_major_ticks[-1]:
                            axes_stats[2].plot(np.sort(xvals), weights, color=color, marker='.', label=ic_per_tick)
                            axes_stats[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        else:
                            axes_stats[2].plot(np.sort(xvals), weights, color=color, marker='.')

                for k in range(len(xy[l]['x'])):
                    axes_params[j].errorbar(xy[l]['x'][k], gv.mean(xy[l]['y'][k]), xerr=0, yerr=gv.sdev(xy[l]['y'][k]),
                        color=xy[l]['color'][k], marker='o', mec='white', ls='', label=xy[l]['label'][k])
                    
            for k in range(len(xy[labels[0]]['x'])):
                axes_stats[0].plot(xy[labels[0]]['x'][k], xy[labels[0]]['q'][k], marker='.', ls='', color=xy[labels[0]]['color'][k])
                axes_stats[1].vlines(xy[labels[0]]['x'][k], ymin=1, ymax=xy[labels[0]]['chi2r'][k], color=xy[labels[0]]['color'][k])
                axes_stats[1].plot(xy[labels[0]]['x'][k], xy[labels[0]]['chi2r'][k], marker='.', ls='', color=xy[labels[0]]['color'][k])

            # format axes
            axes_stats[0].set_ylabel('$Q$')
            axes_stats[0].set_ylim(-0.05, 1.05)
            axes_stats[1].set_ylabel(r'$\chi^2_\nu$')
            axes_stats[1].set_ylim(-0.05, 2.05)
            if ic_per_tick is not None:
                axes_stats[2].set_ylabel('prob')
            if xlabel is None:
                axes[-1].set_xlabel(major_ticks)
            else:
                axes[-1].set_xlabel(xlabel)
            axes[-1].set_xlim(x_major_ticks[0]-0.5, x_major_ticks[-1]+0.5)

            for ax in np.array(axes).flatten():
                for xm, key_val in zip(range(x_major_ticks[0], x_major_ticks[-1]), varied_keys[major_ticks]):
                    ax.axvline(xm-0.5, linestyle='--', alpha=0.1)
                    ax.axvline(xm+0.5, linestyle='--', alpha=0.1)
                    if fit_args_emphasis is not None and key_val == fit_args_emphasis.get_from_path(major_ticks, part_src_snk=part_src_snk):
                        ax.axvline(xm-0.5, linestyle='--', alpha=0.8)
                        ax.axvline(xm+0.5, linestyle='--', alpha=0.8)

                    ax.xaxis.set_ticks_position('none') 

            # sort labels
            hands, labs = axes[0].get_legend_handles_labels()
            labels_dict = dict(zip(labs, hands))
            labels_band = []
            if best_vals is not None and fit_args_emphasis in fit_args_list and show_best:
                labels_band.append('Best')
            if show_avg:
                labels_band.append(ic)

            # Taken from https://stackoverflow.com/a/16090640
            nat_sort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

            if show_legend:
                axes[0].legend(
                    [labels_dict[k] for k in labels_band] + [labels_dict[k] for k in sorted(list(labels_dict), key=nat_sort) if k not in labels_band], 
                    [k for k in labels_band] + [k for k in sorted(list(labels_dict), key=nat_sort) if k not in labels_band], 
                    loc='center left', bbox_to_anchor=(1, 0.5))

            
            if ncols > 1:
            
                gs = matplotlib.gridspec.GridSpec(int(np.ceil(len(labels)/ncols)) + 1, ncols)
                for j, ax in enumerate(axes_params):
                    ax.set_position(gs[j].get_position(fig))

                size = fig.get_size_inches()
                fig.set_size_inches(size[0] *(3 + 4*ncols)/7, size[1])

            plt.close()
            
            return fig
