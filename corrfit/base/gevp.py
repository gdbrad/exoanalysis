import numpy as np
import gvar as gv
import scipy
import functools
import matplotlib
import matplotlib.pyplot as plt

from corrfit.plot import default_cmap
from corrfit.io import bin_data
import corrfit.base.resample


class GEVP(object):

    def __init__(self, raw_correlators, rw_factors=None, gevp_key=None, off_diagonal_key=None, 
            t0=None, td=None, jackknife=True, n_copies=None, seed=None, bin_size=None):
        if gevp_key is None:
            gevp_key = list(raw_correlators)[0][0]

        if gevp_key not in [p_ss[0] for p_ss in raw_correlators]:
            raise ValueError(str(gevp_key)+": not in raw_correlators")
        
        smearings = np.unique([src for part, (src, snk) in raw_correlators if part == gevp_key])
        max_states = len(smearings)

        if bin_size is not None and bin_size != 1:
            raw_correlators = {k : bin_data(d, bin_size=bin_size) for k, d in raw_correlators.items()}
            if rw_factors is not None:
                rw_factors = bin_data(rw_factors, bin_size=bin_size)
        
        if jackknife:
            self.resampler = corrfit.base.resample.JackknifedData(
                data=raw_correlators, rw_factors=rw_factors)
        else:
            self.resampler = corrfit.base.resample.BootstrappedData(
                data=raw_correlators, rw_factors=rw_factors, n_copies=n_copies, seed=seed)

        self.jackknife = jackknife
        self.bin_size = bin_size
        self.raw_correlators = raw_correlators
        self.max_states = max_states
        self.smearings = smearings
        self.gevp_key = gevp_key
        self.off_diagonal_key = off_diagonal_key
        self.t0 = t0
        self.td = td


    def _dict_to_array(self, input_data):
        '''Converts a dictionary of gvars/arrays into an array of gvars/arrays
        Assumes entries of dict are tuples of integers w/ part identifier,
        e.g. (0, 0) ... (N, N)
        '''
        array_shape = input_data[list(input_data)[0]].shape
        num_corrs = (
            + np.max([int(src) for part, (src, snk) in input_data]) 
            - np.min([int(src) for part, (src, snk) in input_data]) 
            + 1)

        output = np.zeros(
            (array_shape + (num_corrs, num_corrs)), 
            dtype=input_data[list(input_data)[0]].dtype)

        # if input is a dictionary of gvar objects
        if len(output.shape) == 3:
            for (_, (src, snk)), corr in input_data.items():
                output[:, src, snk] = corr
        # if input is a dictionary of unaveraged correlators
        elif len(output.shape) == 4:
            for (_, (src, snk)), corr in input_data.items():
                output[:, :, src, snk] = corr
        else: 
            return None

        return output


    def eig(self, corr_array, t0, td):
        """
        Solve the GEVP on correlator matrix `corr_array`
        with GEVP parameters `t0`, `td`
        """
        if t0 is None or td is None:
            raise ValueError("Must specify t0, td")
        
        try:
            vals, vecs = scipy.linalg.eigh(corr_array[td, :, :], corr_array[t0, :, :], eigvals_only=False)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError("Solving GEVP failed for t0, td = (%s, %s)\nTry setting t_max = td - 4"%(t0, td))
        vecs = vecs.T # now row i contains ith eigenvalue

        # sort by eigenvalues --
        # everything else is too slow
        idx = np.argsort(-vals)

        eigenvectors = []
        for v in vecs:
            norm = np.einsum('i,ij,j->', v.conj(), corr_array[t0], v) 
            #eigenvectors.append(v)
            eigenvectors.append(v/norm)
        eigenvectors = np.array(eigenvectors)
        
        return vals[idx], eigenvectors[idx]
    

    @functools.lru_cache()
    def get_eigens(self, t, t0=None, td=None, vary='td'):
        if vary == 't0' and td is None:
            raise ValueError('Must specify td when varying t0')
        elif vary == 'td' and t0 is None:
            raise ValueError('Must specify t0 when varying td')
        elif vary not in ['t0', 'td', 'both']:
            raise ValueError("vary must be in ['t0', 'td', 'both]")
        
        t = np.array(t)
        evalues = np.zeros((self.resampler.n_copies+1, len(t), self.max_states))
        evectors = np.zeros((self.resampler.n_copies+1, len(t), self.max_states, self.max_states))
        for j, data_rs in enumerate(self.resampler.resample(means_only=True)):
            corr_gevp = self._dict_to_array({p_ss : data_rs[p_ss] for p_ss in data_rs if p_ss[0] == self.gevp_key})

            vals = []
            vecs = []
            for ti in t:
                if vary == 't0':
                    va, ve = self.eig(corr_gevp, t0=ti, td=td)
                elif vary == 'td':
                    va, ve = self.eig(corr_gevp, t0=t0, td=ti)
                elif vary == 'both':
                    va, ve = self.eig(corr_gevp, t0=int((ti+1)/2), td=ti)

                vals.append(va)
                vecs.append(ve)
            
            evalues[j] = np.array(vals)
            evectors[j] = np.array(vecs)

        return evalues, evectors
    

    def plot_eigenvalues(self, t=None, t0=None, td=None, vary=None, ylim=None, t_plot_min=None, t_plot_max=None):
        if t0 is None:
            t0 = self.t0
        if td is None:
            td = self.td

        if vary is None:
            if t0 is None:
                vary = 'both'
            else:
                vary = 'td'

        if t is None:
            if vary == 't0':
                t = range(0, td)
            elif vary == 'td':
                rep_key = next((p, ss) for p, ss in self.raw_correlators if p == self.gevp_key)
                t = range(t0, self.raw_correlators[rep_key].shape[1]-1)
            elif vary == 'both':
                rep_key = next((p, ss) for p, ss in self.raw_correlators if p == self.gevp_key)
                t = range(0, self.raw_correlators[rep_key].shape[1]-1)

        evalues, _ = self.get_eigens(t=t, t0=t0, td=td, vary=vary)
        #evalues = gv.dataset.avg_data(evalues, bstrap=True)
        evalues = corrfit.io.to_gvar(evalues, preprocessed=True, jackknife=self.jackknife, bootstrap=(not self.jackknife))
        t = np.array(t)

        fig, ax = plt.subplots()
        for j in range(len(evalues[0])):
            dx = (j+1)/(len(evalues[0])+1) - 0.5
            color = default_cmap((j+1)/(len(evalues[0])+1))
            ax.errorbar(x=t+dx/2, xerr=None, y=[gv.mean(v[j]) for v in evalues], yerr=[gv.sdev(v[j]) for v in evalues], 
                ls='', marker='o', color=color, markeredgecolor='k', label=str(j))

        valid_times = [np.nan if all(v is None for v in evalues[j]) else t for j, t in enumerate(t)]
        if t_plot_min is None or t_plot_min < np.nanmin(valid_times):
            t_plot_min = np.nanmin(valid_times)
        
        if t_plot_max is None or np.nanmax(valid_times) < t_plot_max:
            t_plot_max = np.nanmax(valid_times)

        if t_plot_max < t_plot_min:
            t_plot_min, t_plot_max = t_plot_max, t_plot_min

        for ti in np.arange(t_plot_min, t_plot_max+1):
            ax.axvline(ti-0.5, linestyle='--', alpha=0.1)
            ax.axvline(ti+0.5, linestyle='--', alpha=0.1)
        ax.set_xlim(t_plot_min-1, t_plot_max)

        if ylim is not None: 
            ax.set_ylim(ylim)

        ax.set_yscale('log')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        if vary == 't0':
            ax.set_xlabel('$t_0/a$')
            if t0 is not None:
                ax.axvline(t0-0.5, linestyle='--', alpha=0.1)
                ax.axvline(t0+0.5, linestyle='--', alpha=0.1)
                ax.set_ylabel(r'$\lambda({t_d}, t_0)$'.format(t_d=td))
            else:
                ax.set_ylabel(r'$\lambda$')
        elif vary == 'td':
            ax.set_xlabel('$t_d/a$')
            if td is not None:
                ax.axvline(td-0.5, linestyle='--', alpha=0.6)
                ax.axvline(td+0.5, linestyle='--', alpha=0.6)
                ax.set_ylabel(r'$\lambda(t_d, {t_0})$'.format(t_0=t0))
            else:
                ax.set_ylabel(r'$\lambda$')
        elif vary == 'both':
            ax.set_xlabel('$t/a$')
            ax.set_ylabel(r'$\lambda(t, \lceil t/2 \rceil)$')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.close()
        return fig
    

    def plot_eigenvectors(self, t=None, t0=None, td=None,  vary=None, ylim=None, t_plot_min=None, t_plot_max=None):
        
        norm = lambda v : np.sqrt(np.sum([vi**2 for vi in v]))
        if t0 is None:
            t0 = self.t0
        if td is None:
            td = self.td

        if vary is None:
            if t0 is None:
                vary = 'both'
            else:
                vary = 'td'

        if t is None:
            if vary == 't0':
                t = range(0, td)
            elif vary == 'td':
                rep_key = next((p, ss) for p, ss in self.raw_correlators if p == self.gevp_key)
                t = range(t0, self.raw_correlators[rep_key].shape[1]-1)
            elif vary == 'both':
                rep_key = next((p, ss) for p, ss in self.raw_correlators if p == self.gevp_key)
                t = range(0, self.raw_correlators[rep_key].shape[1]-1)

        _, evectors = self.get_eigens(t=t, t0=t0, td=td, vary=vary)
        #evectors = gv.dataset.avg_data(np.abs(evectors), bstrap=True)
        evectors = corrfit.io.to_gvar(np.abs(evectors), preprocessed=True, jackknife=self.jackknife, bootstrap=(not self.jackknife))
        t = np.array(t)

        fig, axes = plt.subplots(nrows=evectors.shape[1],
            sharex=True, sharey=True, gridspec_kw={'hspace':0.1})
        size = fig.get_size_inches()
        fig.set_size_inches(size[0], size[1]*(3 + 4*evectors.shape[1])/7)

        #cmap = matplotlib.cm.get_cmap('gist_ncar_r')
        for j in range(evectors.shape[1]):

            dx = (j+1)/(evectors.shape[1]+1) - 0.5
            color = default_cmap((j+1)/(evectors.shape[1]+1))
            for k in range(evectors.shape[2]):
                if k == 0:
                    label = str(j)
                else:
                    label = None
                axes[k].errorbar(x=t+dx/2, xerr=None, y=[gv.mean(v.T[j][k] / norm(v.T[j]) )  for v in evectors], yerr=[gv.sdev(v.T[j][k] / norm(v.T[j])) for v in evectors], 
                    ls='--', marker='.', color=color, markeredgecolor='k', label=label)

        valid_times = [np.nan if all(v is None for v in evectors[j]) else t for j, t in enumerate(t)]
        if t_plot_min is None or t_plot_min < np.nanmin(valid_times):
            t_plot_min = np.nanmin(valid_times)
        
        if t_plot_max is None or np.nanmax(valid_times) < t_plot_max:
            t_plot_max = np.nanmax(valid_times)

        if t_plot_max < t_plot_min:
            t_plot_min, t_plot_max = t_plot_max, t_plot_min

        for ti in np.arange(t_plot_min, t_plot_max+1):
            for ax in axes:
                ax.axvline(ti-0.5, linestyle='--', alpha=0.1)
                ax.axvline(ti+0.5, linestyle='--', alpha=0.1)
        axes[-1].set_xlim(t_plot_min-1, t_plot_max)
        axes[-1].set_ylim(-0.05, 1.05)
        axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        if vary == 't0':
            axes[-1].set_xlabel('$t_0/a$')
            for k in range(evectors.shape[1]):
                if t0 is not None:
                    axes[k].axvline(t0-0.5, linestyle='--', alpha=0.6)
                    axes[k].axvline(t0+0.5, linestyle='--', alpha=0.6)
                    axes[k].set_ylabel(r'$|\hat v^{k}_i (t_0, {td})|$'.format(k=k, td=td))
                else:
                    axes[k].set_ylabel(r'$|\hat v^{%s}_i|$'%(k))
        elif vary == 'td':
            axes[-1].set_xlabel('$t_d/a$')
            for k in range(evectors.shape[1]):
                if td is not None:
                    axes[k].axvline(td-0.5, linestyle='--', alpha=0.6)
                    axes[k].axvline(td+0.5, linestyle='--', alpha=0.6)
                    axes[k].set_ylabel(r'$|\hat v^{k}_i ({t0}, t_d)|$'.format(k=k, t0=t0))
                else:
                    axes[k].set_ylabel(r'$|\hat v^{%s}_i|$'%(k))
        elif vary == 'both':
            axes[-1].set_xlabel('$t_d/a$')
            for k in range(evectors.shape[1]):
                axes[k].set_ylabel(r'$|\hat v^{%s}_i|$'%(k))

        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.close()
        return fig