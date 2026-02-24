import numpy as np
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt
import lsqfit
import scipy
import itertools
import functools
import corrfit.two_pt
import copy
import tqdm

class GEVP(object):

    def __init__(self, raw_correlators, gevp_key=None, t0=None, td=None):
        if gevp_key is None:
            gevp_key = list(raw_correlators)[0][0]

        raw_correlators_gevp = {(part, src_snk) : raw_correlators[(part, src_snk) ]
            for part, src_snk in raw_correlators if part == gevp_key}
        raw_correlators_passthrough = {(part, src_snk) : raw_correlators[(part, src_snk) ]
            for part, src_snk in raw_correlators if part != gevp_key}
        
        smearings = np.unique([src for part, (src, snk) in raw_correlators_gevp])
        max_states = len(smearings)
        
        correlators = gv.dataset.avg_data(raw_correlators)
        averaged_correlators_gevp = {p_ss : correlators[p_ss] for p_ss in raw_correlators_gevp}
        averaged_correlators_passthrough = {p_ss : correlators[p_ss] for p_ss in raw_correlators_passthrough}

        self.raw_correlators_gevp = raw_correlators_gevp
        self.raw_correlators_passthrough = raw_correlators_passthrough
        self.t0 = t0
        self.td = td
        self.max_states = max_states
        self.smearings = np.unique([src for part, (src, snk) in raw_correlators_gevp])
        self.gevp_key = gevp_key
        self.averaged_correlators_gevp = averaged_correlators_gevp
        self.averaged_correlators_passthrough = averaged_correlators_passthrough
            

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
            for (part, (src, snk)), corr in input_data.items():
                output[:, src, snk] = corr
        # if input is a dictionary of unaveraged correlators
        elif len(output.shape) == 4:
            for (part, (src, snk)), corr in input_data.items():
                output[:, :, src, snk] = corr
        else: 
            return None

        return output


    def _set_defaults(self, t0=None, td=None, max_states=None):
        if ((t0 is None and self.t0 is None) 
            or (td is None and self.td is None)
        ):
            raise ValueError('Must specify t0 and td.')
        
        if t0 is None:
            t0 = self.t0
        if td is None:
            td = self.td
        if max_states is None:
            max_states = self.max_states

        return t0, td, max_states



    @functools.cache
    def eig(self, t0=None, td=None, max_states=None, estimate_errors=False, t_ref=None, sorting_algorithm=None):
        '''
        Returns eigenvalues, eigenvectors of covariance matrix, sorted by smallest eigenvalue
        Note: if returned as gvar, imaginary entries will be lost!
        '''

        t0, td, max_states = self._set_defaults(t0, td, max_states)

        # sort eigenvalues, eigenvectors by effective mass
        corr_array = np.mean(self._dict_to_array(self.raw_correlators_gevp), axis=0)
        #vals0 = scipy.linalg.eigh(corr_array[td, :, :], corr_array[t0, :, :], eigvals_only=True)
        #vals1 = scipy.linalg.eigh(corr_array[td, :, :], corr_array[t0+1, :, :], eigvals_only=True)

        #eff_mass = -np.log(vals0/vals1)
        
        #idx = np.argsort(eff_mass)
        #idx = np.argsort(-vals0)
        #print(eff_mass[idx])


        if estimate_errors:
            temp_array = self._dict_to_array(self.averaged_correlators_gevp)

            u, d, ut = gv.linalg.svd(temp_array[t0, :, :].real)
            corr_sqrt = (u @ np.diag(np.sqrt(d)) @ ut) 
            matrix = gv.linalg.inv(corr_sqrt) @ temp_array[td] @ gv.linalg.inv(corr_sqrt)

            #vals, vecs = gv.linalg.eigvalsh(matrix, eigvec=True)
            vals_gvar = gv.linalg.eigvalsh(matrix, eigvec=False)
            #_, vecs = scipy.linalg.eigh(corr_array[td, :, :], corr_array[t0, :, :], eigvals_only=False)

        vals, vecs = scipy.linalg.eigh(corr_array[td, :, :], corr_array[t0, :, :], eigvals_only=False)
        vecs = vecs.T # now row i contains ith eigenvalue

        #idx = self.sort_states(eigenvalues=vals, sorting_algorithm='eigenvalues')
        #vals, vecs = vals[idx], vecs[idx]

        idx = self.sort_states(eigenvalues=vals, eigenvectors=vecs, t0=t0, t_ref=t_ref, sorting_algorithm=sorting_algorithm)

        eigenvectors = []
        for v in vecs:
            norm = np.einsum('i,ij,j->', v.conj(), corr_array[t0], v) 
            #eigenvectors.append(v)
            eigenvectors.append(v/norm)
        eigenvectors = np.array(eigenvectors)

        if estimate_errors:
            vals = vals_gvar - gv.mean(vals_gvar) + vals
        
        return vals[idx][:max_states], eigenvectors[idx][:max_states]
        #return gv.gvar(vals[idx], 0.01 *vals[idx]), eigenvectors[idx]


    def sort_states(self, eigenvalues=None, eigenvectors=None, t0=None, t_ref=None, sorting_algorithm=None):
        if sorting_algorithm is None:
            sorting_algorithm = 'eigenvalues'

        if t0 is None:
            t0 = self.t0

        if t_ref is None:
            t_ref = t0 + 1

        if sorting_algorithm == 'eigenvalues':
            return np.argsort(-eigenvalues)
        
        elif sorting_algorithm == 'eigenvectors-old':
            vals_ref, vecs_ref = self.eig(t0=t0, td=t_ref, sorting_algorithm='eigenvalues')
            idx_ref = self.sort_states(eigenvalues=vals_ref, sorting_algorithm='eigenvalues')
            vecs_ref = np.array(vecs_ref[idx_ref])

            permutations = np.array(list(itertools.permutations(idx_ref)))
            temp_matrix = np.stack([eigenvectors[permut] for permut in permutations])
            idx = np.argmax(np.prod(np.abs(np.einsum('ij,lij->li', vecs_ref, temp_matrix)), axis=1))
            
            return permutations[idx]
        
        # same as above, except calculates scalar product using complex conjugate
        elif sorting_algorithm == 'eigenvectors':
            vals_ref, vecs_ref = self.eig(t0=t0, td=t_ref, sorting_algorithm='eigenvalues')
            idx_ref = self.sort_states(eigenvalues=vals_ref, sorting_algorithm='eigenvalues')
            vecs_ref = np.array(vecs_ref[idx_ref])

            permutations = np.array(list(itertools.permutations(idx_ref)))
            temp_matrix = np.stack([eigenvectors[permut] for permut in permutations])
            idx = np.argmax(np.prod(np.abs(np.einsum('ij,lij->li', np.conj(vecs_ref), temp_matrix)), axis=1))
            
            return permutations[idx]
        
        elif sorting_algorithm == 'eigenvectors-recursive':
            return None # doesn't work
            if t_ref < t0 + 2:
                return np.argsort(-eigenvalues)
            else:
                vals_ref, vecs_ref = self.eig(t0=t0, td=t_ref-1)
                idx_ref = self.sort_states(eigenvalues=vals_ref, eigenvectors=vecs_ref, sorting_algorithm='eigenvectors-recursive', t_ref=t_ref-1)
                vecs_ref = np.array(vecs_ref[idx_ref])

                permutations = np.array(list(itertools.permutations(idx_ref)))
                temp_matrix = np.stack([eigenvectors[permut] for permut in permutations])
                idx = np.argmax(np.prod(np.abs(np.einsum('ij,lij->li', vecs_ref, temp_matrix)), axis=1))

                return permutations[idx]
        
        elif sorting_algorithm == 'volume':
            return None
        

    def optimized_correlators(self, 
            corr_type='rotated', ratio=False, ratio_particles=None,
            t0=None, td=None, max_states=None, t_ref=None, sorting_algorithm=None):

        if corr_type == 'rotated':
            correlators = self._diagonalize(
                t0=t0, td=td, max_states=max_states, 
                sorting_algorithm=sorting_algorithm, t_ref=t_ref, 
                diagonal_only=True, as_gvar=(not ratio))
        
        elif corr_type == 'principal':
            correlators = self._get_principal_correlators(
                t0=t0, max_states=max_states, 
                sorting_algorithm=sorting_algorithm, t_ref=t_ref)

        else:
            raise ValueError('Improper corr_type')
        
        if ratio:
            if ratio_particles is None:
                raise ValueError('Must specify keys for denominator of ratio fit')
            correlators = self._make_ratio_correlators(correlators, ratio_particles)

        return correlators


    def _diagonalize(self, t0=None, td=None, max_states=None, diagonal_only=True, t_ref=None, sorting_algorithm=None, as_gvar=True):
        t0, td, max_states = self._set_defaults(t0, td, max_states)
            
        _, eigenvectors = self.eig(t0=t0, td=td, max_states=max_states, estimate_errors=False, t_ref=t_ref, sorting_algorithm=sorting_algorithm)
        corr_array = self._dict_to_array(self.raw_correlators_gevp)
        
        # transpose to make columns eigenvectors 
        eig_matrix = eigenvectors.T 
        #rotated_corr_array = (eig_matrix.T.conj() @ corr_array @ eig_matrix)
        rotated_corr_array = np.einsum('ji,mnjk,kl -> mnil', eig_matrix.conj(), corr_array, eig_matrix)

        # return gvar, object decorrelate diagonal entries
        output = {}
        for p_ss in self.raw_correlators_passthrough:
            output[p_ss] = self.raw_correlators_passthrough[p_ss]

        for part, (src, snk) in self.raw_correlators_gevp:
            if src < max_states and snk < max_states:
                if diagonal_only:
                    if src == snk:
                        #output[(src, snk)] = gv.dataset.avg_data(rotated_corr_array[:, :, src, snk])
                        output[(part, (src, snk))] = rotated_corr_array[:, :, src, snk]
                else:
                    #output[(src, snk)] = gv.dataset.avg_data(rotated_corr_array[:, :, src, snk])
                    output[(part, (src, snk) )] = rotated_corr_array[:, :, src, snk]

        if as_gvar:
            return gv.dataset.avg_data(output)
        else:
            return output


    def _get_principal_correlators(self, t0=None, td=None, max_states=None, t_ref=None, sorting_algorithm=None):
        t0, td, max_states = self._set_defaults(t0, td, max_states)

        t = np.arange(len(self.averaged_correlators_gevp[list(self.averaged_correlators_gevp)[0]]))

        output = {p_ss : self.averaged_correlators_passthrough[p_ss]
            for p_ss in self.averaged_correlators_passthrough}
        for s in self.smearings[:max_states]:
            output[(self.gevp_key, (s, s))] =  np.array([])

        for td in t:
            eigs, _ = self.eig(t0=t0, td=td, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)
            for s, v in zip(self.smearings[:max_states], eigs):
                output[(self.gevp_key, (s,s))] = np.append(output[(self.gevp_key, (s,s))], v)

        return output
    

    def _make_ratio_correlators(self, optimized_correlators, ratio_particles):
        # ratio_particles: particles for denominator
        # eg, [[part1a, part1b], [part2a, part2b], ...]

        def construct_denominator(correlators, keys):
            if len(keys) == 1:
                return correlators[keys[0]]**2
            elif len(keys) == 2:
                return correlators[keys[0]] *correlators[keys[1]]
            else:
                print('Warning: possibly too many keys when constructing denominator')
                return (np.prod([correlators[k] for k in keys]))**(2/len(keys))

        input_is_gvar = bool(len(optimized_correlators[list(optimized_correlators)[0]].shape) == 1)
        
        den_part_srcsnks = [[next(p_ss for p_ss in self.averaged_correlators_passthrough if p_ss[0] == part)
                              for part in part_list] for part_list in ratio_particles]

        if input_is_gvar:
            correlators = {}
            correlators.update({p_ss : self.averaged_correlators_passthrough[p_ss]
                    for p_ss in self.averaged_correlators_passthrough})
            correlators.update(optimized_correlators)
            
        else:
            raw_correlators = {}
            raw_correlators.update({p_ss : self.raw_correlators_passthrough[p_ss]
                for p_ss in self.raw_correlators_passthrough})
            raw_correlators.update(optimized_correlators)
            correlators = gv.dataset.avg_data(raw_correlators)

        for j, part_srcsnk_list in enumerate(den_part_srcsnks):
            correlators[(self.gevp_key, (j, j))] = correlators[(self.gevp_key, (j, j))] / construct_denominator(correlators, part_srcsnk_list)

        return correlators


    def plot_eigenvalues(self, t0=None, td=None, max_states=None, vary='td', ylim=None, t_plot_min=None, t_plot_max=None, t_ref=None, sorting_algorithm=None):
        t0, td, max_states = self._set_defaults(t0, td, max_states)

        if vary not in ['t0', 'td']:
            raise ValueError("vary must be in ['t0', 'td']")
        elif vary == 't0':
            t = np.arange(0, td)
        elif vary == 'td':
            t = np.arange(t0, len(self.averaged_correlators_gevp[list(self.averaged_correlators_gevp)[0]])-1)

        values = []
        for ti in t:
            try:
                if vary == 't0':
                    values.append(self.eig(t0=ti, td=td, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)[0])
                elif vary == 'td':
                    values.append(self.eig(t0=t0, td=ti, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)[0])
            except ValueError:
                values.append(np.repeat(None, np.sqrt(len(self.raw_correlators_gevp))))

        fig, ax = plt.subplots()
        # create colormap from 'cool' and 'winter'
        colors1 = matplotlib.cm.get_cmap('cool_r')(np.linspace(0., 1, 128))
        colors2 = matplotlib.cm.get_cmap('winter_r')(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

        #cmap = matplotlib.cm.get_cmap('gist_ncar_r')
        for j in range(len(values[0])):
            dx = (j+1)/(len(values[0])+1) - 0.5
            color = cmap((j+1)/(len(values[0])+1))
            ax.errorbar(x=t+dx/2, xerr=None, y=[gv.mean(v[j]) for v in values], yerr=[gv.sdev(v[j]) for v in values], 
                ls='', marker='o', color=color, markeredgecolor='k')

        valid_times = [np.nan if all(v is None for v in values[j]) else t for j, t in enumerate(t)]
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
                ax.set_ylabel('$\lambda({t_d}, t_0)$'.format(t_d=td))
            else:
                ax.set_ylabel('$\lambda$')
        elif vary == 'td':
            ax.set_xlabel('$t_d/a$')
            if td is not None:
                ax.axvline(td-0.5, linestyle='--', alpha=0.6)
                ax.axvline(td+0.5, linestyle='--', alpha=0.6)
                ax.set_ylabel('$\lambda(t_d, {t_0})$'.format(t_0=t0))
            else:
                ax.set_ylabel('$\lambda$')

        plt.close()
        return fig


    def plot_eigenvectors(self, t0=None, td=None, max_states=None, vary='td', ylim=None, t_plot_min=None, t_plot_max=None, t_ref=None, sorting_algorithm=None):
        t0, td, max_states = self._set_defaults(t0, td, max_states)

        if vary not in ['t0', 'td']:
            raise ValueError("vary must be in ['t0', 'td']")
        elif vary == 't0':
            t = np.arange(0, td)
        elif vary == 'td':
            t = np.arange(t0, len(self.averaged_correlators_gevp[list(self.averaged_correlators_gevp)[0]])-1)

        values = []
        for ti in t:
            try:
                if vary == 't0':
                    values.append(self.eig(t0=ti, td=td, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)[1])
                elif vary == 'td':
                    values.append(self.eig(t0=t0, td=ti, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)[1])
            except ValueError:
                values.append(np.repeat(None, np.sqrt(len(self.raw_correlators_gevp))))

        values = np.abs(np.array(values))

        fig, axes = plt.subplots(nrows=values.shape[1],
            sharex=True, gridspec_kw={'hspace':0.1})
        size = fig.get_size_inches()
        fig.set_size_inches(size[0], size[1]*(3 + 4*values.shape[1])/7)

        # create colormap from 'cool' and 'winter'
        colors1 = matplotlib.cm.get_cmap('cool_r')(np.linspace(0., 1, 128))
        colors2 = matplotlib.cm.get_cmap('winter_r')(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

        #cmap = matplotlib.cm.get_cmap('gist_ncar_r')
        for j in range(len(values[0])):
            dx = (j+1)/(len(values[0])+1) - 0.5
            color = cmap((j+1)/(len(values[0])+1))
            for k in range(values.shape[1]):
                axes[k].errorbar(x=t+dx/2, xerr=None, y=[gv.mean(v[j][k]) for v in values], yerr=[gv.sdev(v[j][k]) for v in values], 
                    ls='--', marker='.', color=color, markeredgecolor='k')

        valid_times = [np.nan if all(v is None for v in values[j]) else t for j, t in enumerate(t)]
        if t_plot_min is None or t_plot_min < np.nanmin(valid_times):
            t_plot_min = np.nanmin(valid_times)
        
        if t_plot_max is None or np.nanmax(valid_times) < t_plot_max:
            t_plot_max = np.nanmax(valid_times)

        if t_plot_max < t_plot_min:
            t_plot_min, t_plot_max = t_plot_max, t_plot_min

        for ti in np.arange(t_plot_min, t_plot_max+1):
            pass
            #ax.axvline(ti-0.5, linestyle='--', alpha=0.1)
            #ax.axvline(ti+0.5, linestyle='--', alpha=0.1)
        axes[-1].set_xlim(t_plot_min-1, t_plot_max)

        #ax.set_yscale('log')
        axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        if vary == 't0':
            axes[-1].set_xlabel('$t_0/a$')
            for k in range(values.shape[1]):
                if t0 is not None:
                    axes[k].axvline(t0-0.5, linestyle='--', alpha=0.6)
                    axes[k].axvline(t0+0.5, linestyle='--', alpha=0.6)
                    axes[k].set_ylabel(r'$|v^{k}_i (t_0, {td})|$'.format(k=k, td=td))
                else:
                    axes[k].set_ylabel(r'$|v^{%s}_i|$'%(k))
        elif vary == 'td':
            axes[-1].set_xlabel('$t_d/a$')
            for k in range(values.shape[1]):
                if td is not None:
                    axes[k].axvline(td-0.5, linestyle='--', alpha=0.6)
                    axes[k].axvline(td+0.5, linestyle='--', alpha=0.6)
                    axes[k].set_ylabel(r'$|v^{k}_i ({t0}, t_d)|$'.format(k=k, t0=t0))
                else:
                    axes[k].set_ylabel(r'$|v^{%s}_i|$'%(k))

        plt.close()
        return fig

    def plot_principal_effective_mass(self, t0=None, td=None, max_states=None, vary='td', ylim=None, t_plot_min=None, t_plot_max=None, t_ref=None, sorting_algorithm=None):
        t0, td, max_states = self._set_defaults(t0, td, max_states)

        if vary not in ['t0', 'td']:
            raise ValueError("vary must be in ['t0', 'td']")
        elif vary == 't0':
            t = np.arange(0, td)
        elif vary == 'td':
            t = np.arange(t0, len(self.averaged_correlators_gevp[list(self.averaged_correlators_gevp)[0]])-1)

        values = []
        for ti in t:
            try:
                if vary == 't0':
                    values.append(self.eig(t0=ti, td=td, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)[0])
                elif vary == 'td':
                    values.append(self.eig(t0=t0, td=ti, max_states=max_states, estimate_errors=True, t_ref=t_ref, sorting_algorithm=sorting_algorithm)[0])
            except ValueError as e:
                print(e)
                values.append(np.repeat(None, np.sqrt(len(self.raw_correlators_gevp))))

        # effective mass
        t = t[:-1]
        values = np.array(values)[:-1]
        values = np.log(values / np.roll(values, -1, axis=0))
        if vary == 't0':
            values = -values

        fig, ax = plt.subplots()
        
        # create colormap from 'cool' and 'winter'
        colors1 = matplotlib.cm.get_cmap('cool_r')(np.linspace(0., 1, 128))
        colors2 = matplotlib.cm.get_cmap('winter_r')(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

        for j in range(len(values[0])):
            dx = (j+1)/(len(values[0])+1) - 0.5
            color = cmap((j+1)/(len(values[0])+1))
            ax.errorbar(x=t+dx/2, xerr=None, y=[gv.mean(v[j]) for v in values], yerr=[gv.sdev(v[j]) for v in values], 
                ls='', marker='o', color=color, markeredgecolor='k')

        valid_times = [np.nan if all(v is None for v in values[j]) else t for j, t in enumerate(t)]
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

        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        if vary == 't0':
            ax.set_xlabel('$t_0/a$')
            if t0 is not None:
                ax.axvline(t0-0.5, linestyle='--', alpha=0.6)
                ax.axvline(t0+0.5, linestyle='--', alpha=0.6)
                ax.set_ylabel(r'$a E^{\rm eff} [t_d=%s]$'%(td))
            else:
                ax.set_ylabel(r'$a E^{\rm eff}$')
        elif vary == 'td':
            ax.set_xlabel('$t_d/a$')
            if td is not None:
                ax.axvline(td-0.5, linestyle='--', alpha=0.6)
                ax.axvline(td+0.5, linestyle='--', alpha=0.6)
                ax.set_ylabel(r'$a E^{\rm eff} [t_0=%s]$'%(t0))
            else:
                ax.set_ylabel(r'$a E^{\rm eff}$')

        if ylim is None:
            min, max = gv.gvar(np.inf, 0), gv.gvar(0,0)
            for v in values.T:
                avg = lsqfit.wavg(v[~np.isnan(gv.mean(v))][:-2])
                #print(v, avg)
                #print(avg)
                min = np.nanmin(np.append(avg, min))
                max = np.nanmax(np.append(avg, max))

            dy = max - min
            ax.set_ylim(gv.mean(min)- 0.3 *gv.mean(dy), gv.mean(max) + 0.3 *gv.mean(dy))
        else:
            ax.set_ylim(ylim)

        plt.close()
        return fig
    

class FitManager(corrfit.two_pt.FitManager):

    def __init__(self, raw_correlators, prior, fit_args, gevp_args, gevp_key):
        gevp = GEVP(raw_correlators=raw_correlators,
                    gevp_key=gevp_key,
                    t0=gevp_args['t0'], 
                    td=gevp_args['td'],)
                    #max_states=gevp_args['n_states'])
        
        correlators = gevp.optimized_correlators(
            corr_type=gevp_args['corr_type'],
            ratio=bool(gevp_args['fit_type'] == 'ratio'),
            ratio_particles=gevp_args['ratio_den'] if 'ratio_den' in gevp_args else None,
            t0=gevp_args['t0'],
            td=gevp_args['td'],
            max_states=gevp_args['n_states'],
            sorting_algorithm=gevp_args['sorting']
        )

        # method for automatically adjusting prior -- refactor later
        if ('ratio_den' in gevp_args 
                and gevp_args['ratio_den'] is not None
                and set([i for s in gevp_args['ratio_den'] for i in s]) <= set([p for p, _ in correlators])):
            
            ratio_particles = gevp_args['ratio_den']

            # fit denominator only to obtain posterior
            correlator_den = {p_ss : correlators[p_ss] 
                for p_ss in correlators if p_ss[0] in [i for s in gevp_args['ratio_den'] for i in s]}
            fit_args_den = corrfit.two_pt.FitArgs(
                data=correlator_den,
                fit_args={part : fit_args[part]
                    for part in fit_args if part in [i for s in gevp_args['ratio_den'] for i in s]}
            )
            fitter_den = corrfit.fitters.TwoPtFitter(
                data=correlator_den,
                prior=prior,
                fit_args=fit_args_den
            )
            posterior_den = fitter_den.fit.p
            #print(gv.tabulate(posterior_den))

            # only need p_keys, not the actual fit of numerator
            correlator_num = {p_ss : correlators[p_ss] 
                for p_ss in correlators if p_ss[0] == gevp_key}
            fit_args_num = corrfit.two_pt.FitArgs(
                data=correlator_num,
                fit_args={part : fit_args[part]
                    for part in fit_args if part == gevp_key}
            )
            num_p_keys = corrfit.fitters.TwoPtFitter(
                data=correlator_num,
                prior=prior,
                fit_args=fit_args_num
            ).p_keys

            den_part_srcsnks = [[next(p_ss for p_ss in list(fitter_den.p_keys) if p_ss[0] == part)
                for part in part_list] for part_list in ratio_particles]

            #print(den_part_srcsnks)
            #print(num_p_keys)

            new_prior = {}
            new_prior.update(prior)

            for j, part_srcsnk_list in enumerate(den_part_srcsnks):
                # interaction energy = gevp energy - 2 average denominator energies
                new_prior[num_p_keys[(gevp_key, (j, j))]['E0']] = (
                    new_prior[num_p_keys[(gevp_key, (j, j))]['E0']] 
                    - (2 / len(part_srcsnk_list)) *np.sum([posterior_den[fitter_den.p_keys[dp_ss]['E0']] for dp_ss in part_srcsnk_list])
                )

                if 'Z_src' in num_p_keys[(gevp_key, (j, j))]:
                    new_prior[num_p_keys[(gevp_key, (j, j))]['Z_src']] = (
                        new_prior[num_p_keys[(gevp_key, (j, j))]['Z_src']] 
                        /(np.prod([posterior_den[fitter_den.p_keys[dp_ss]['Z_src']][0] for dp_ss in part_srcsnk_list])**(2/len(part_srcsnk_list)))
                    )

                if ('Z_snk' in num_p_keys[(gevp_key, (j, j))] 
                    and num_p_keys[(gevp_key, (j, j))]['Z_src'] != num_p_keys[(gevp_key, (j, j))]['Z_snk']):
                    new_prior[num_p_keys[(gevp_key, (j, j))]['Z_snk']] = (
                        new_prior[num_p_keys[(gevp_key, (j, j))]['Z_snk']] 
                        /(np.prod([posterior_den[fitter_den.p_keys[dp_ss]['Z_snk']][0] for dp_ss in part_srcsnk_list])**(2/len(part_srcsnk_list)))
                    )

                if False:
                    pass
                    # implement for wf overlaps!

            prior_ratio = new_prior
        else:
            prior_ratio = None

        super().__init__(correlators=correlators, prior=prior, fit_args=fit_args)

        self.prior_ratio = prior_ratio
        self.gevp_args = gevp_args
        self.gevp_key = gevp_key
        self.gevp = gevp


    @property
    def fit_args(self):
        return FitArgs(self.correlators, self._fit_args, gevp_args=self.gevp_args, gevp_key=self.gevp_key)


    @property
    def fitters(self):
        if self._fitters is None:
            self._fitters = FittersDict(
                gevp=self.gevp, gevp_args=self.gevp_args, prior=self.prior, 
                prior_ratio=self.prior_ratio)
            
        return self._fitters


    # decorator
    # used to iterate over multiple fit_args, 
    # e.g., kwargs = {t_start :  [5, 7]} -> kwargs_list = [{t_start : 5}, {t_start : 7}]
    def iterate_fit_args(func):

        def inner(self, **kwargs):
            kwargs.setdefault('particles', None)
            fargs_keys = ['n_states', 't_start', 't_end', 'energy_gaps', 'svdcut', 'prior_En',
                't0', 'td', 'corr_type', 'fit_type', 'eig_sorting']
            for key in fargs_keys:
                kwargs.setdefault(key, None)

            duplicate_keys = {}
            for key in fargs_keys:
                if key != 'particles':
                    temp = kwargs[key]
                    if temp is not None and not isinstance(temp, str) and hasattr(temp, '__len__'):
                        duplicate_keys[key] = kwargs[key]

            if len(duplicate_keys) == 0:
                return func(self, **kwargs)

            kwargs_list = []
            for temp_prod in itertools.product(*duplicate_keys.values()):
                iter_dict = dict(zip(duplicate_keys.keys(), temp_prod))
                temp = {}
                temp.update(kwargs)
                temp.update(iter_dict)
                kwargs_list.append(temp)

            #return [func(self, **k) for k in kwargs_list]
            return [func(self, **k) for k in tqdm.tqdm(kwargs_list, desc='Collecting FitArgs: ')]

        return inner 


    @iterate_fit_args
    def get_fit_args(self, particles=None, t_start=None, t_end=None, n_states=None, energy_gaps=None, svdcut=None, prior_En=None, 
            t0=None, td=None, corr_type=None, fit_type=None, eig_sorting=None, defaults_only=False):
        
        if particles is None:
            particles = self._get_particles()
        elif particles == 'gevp_key':
            particles = self.gevp_key
        if isinstance(particles, str):
            particles = [particles]

        if fit_type is None:
            fit_type = self.gevp_args['fit_type']

        # always include other particles in fit
        if self.gevp_key in particles: # and fit_type == 'ratio':
            if 'ratio_den' in self.gevp_args and self.gevp_args['ratio_den'] is not None:
                # add ratio denominator particles to 'particles' if doing a ratio fit
                if not set([part for s in self.gevp_args['ratio_den'] for part in s]) <= set(particles):
                    particles = list(set(list(particles) + [part for s in self.gevp_args['ratio_den'] for part in s]))

        # hold fit_args in place for all other particles
        if self.gevp_key in particles:
            fit_args = {part : self.fit_args[part] for part in particles if part != self.gevp_key}
            fit_args.update(super().get_fit_args
                (particles=self.gevp_key, t_start=t_start, t_end=t_end, n_states=n_states, 
                 energy_gaps=energy_gaps, svdcut=svdcut, prior_En=prior_En))

            gevp_args = {}
            gevp_args.update(self.gevp_args)
            gevp_args['t0']             = t0             or gevp_args['t0'] 
            gevp_args['td']             = td             or gevp_args['td'] 
            gevp_args['corr_type']      = corr_type      or gevp_args['corr_type'] 
            gevp_args['fit_type']       = fit_type       or gevp_args['fit_type'] 
            gevp_args['sorting']        = eig_sorting    or gevp_args['sorting'] 

        else:
            fit_args = super().get_fit_args(
                particles=particles, t_start=t_start, t_end=t_end, n_states=n_states, 
                energy_gaps=energy_gaps, svdcut=svdcut, prior_En=prior_En)
            gevp_args = {}

        return corrfit.gevp.FitArgs(self.correlators, fit_args=fit_args, gevp_args=gevp_args, gevp_key=self.gevp_key) 


    @iterate_fit_args
    def get_fits(self, **kwargs):
        fit_args = self.get_fit_args(**kwargs)
        return self.fitters[fit_args].fit


    @iterate_fit_args
    def get_spectrum(self, interaction_only=False, **kwargs):        
        fit_args = self.get_fit_args(**kwargs)
        particles = list(fit_args)
        spectrum = self.fitters[fit_args].spectrum

        if (self.gevp_key in fit_args 
                and fit_args[self.gevp_key]['gevp']['fit_type'] == 'ratio'):
            
            # ratio fits, return interaction energy
            if interaction_only:
                return {(part, src_snk) : spectrum[(part, src_snk)] for part, src_snk in spectrum if part in particles}
            
            # ratio fits, return total energy
            ratio_particles = self.gevp_args['ratio_den']
            den_part_srcsnks = [[next(p_ss for p_ss in list(self.get_p_keys()) if p_ss[0] == part)
                for part in part_list] for part_list in ratio_particles]

            output = {}
            output.update(spectrum)
            for j, part_srcsnk_list in enumerate(den_part_srcsnks):
                output[(self.gevp_key, (j, j))] = (
                    + spectrum[(self.gevp_key, (j, j))]
                    + (2 / len(part_srcsnk_list)) *np.sum([spectrum[dp_ss][0] for dp_ss in part_srcsnk_list])
                )

            return {(part, src_snk) : output[(part, src_snk)] for part, src_snk in output if part in particles}
        else:
            # exp fits, return total energy
            if not interaction_only:
                return {(part, src_snk) : spectrum[(part, src_snk)] for part, src_snk in spectrum if part in particles}
            
            # exp fits, return energy splitting
            ratio_particles = self.gevp_args['ratio_den']
            den_part_srcsnks = [[next(p_ss for p_ss in list(self.get_p_keys()) if p_ss[0] == part)
                for part in part_list] for part_list in ratio_particles]

            output = {}
            output.update(spectrum)
            for j, part_srcsnk_list in enumerate(den_part_srcsnks):
                output[(self.gevp_key, (j, j))] = (
                    + spectrum[(self.gevp_key, (j, j))]
                    - (2 / len(part_srcsnk_list)) *np.sum([spectrum[dp_ss][0] for dp_ss in part_srcsnk_list])
                )

            return {(part, src_snk) : output[(part, src_snk)] for part, src_snk in output if part in particles}
        

    def plot_stability(self, major_ticks=None, param=None, part_src_snk=None, show_all=False, show_avg=True, minor_ticks=None, 
            ylim=None, interaction_only=False, debug=False, **kwargs):
        
        tuple_to_str =  lambda t : t if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'

        
        kwargs['particles'] = kwargs.get('particles') or self.gevp_key
        if isinstance(kwargs['particles'], str):
            kwargs['particles'] = [kwargs['particles']]

        if major_ticks is None:
            major_ticks = 't_start'

        xlabel = None
        if major_ticks == 't_start':
            xlabel = r'$t_{\rm min}$'
        elif major_ticks == 't_end':
            xlabel = r'$t_{\rm max}$'
        elif major_ticks == 't0':
            xlabel = r'$t_0$'
        elif major_ticks == 'td':
            xlabel = r'$t_d$'

        if isinstance(minor_ticks, str):
            minor_ticks = [minor_ticks]
        elif minor_ticks is None:
            minor_ticks = {}
            minor_ticks['t_start'] = r'$t_{\rm min}=$'
            minor_ticks['n_states'] = r'$N=$'
            minor_ticks['gevp/fit_type'] = ''
            minor_ticks['gevp/corr_type'] = ''
            minor_ticks['t_end'] = r'$t_{\rm max}=$'
            minor_ticks['gevp/t0'] = r'$t_0=$'
            minor_ticks['gevp/td'] = r'$t_d=$'
            minor_ticks['gevp/sorting'] = ''
            minor_ticks['prior_En'] = 'Shared $E_{N+1}$: '
            minor_ticks['energy_gaps'] = r'$dE \propto$'
            minor_ticks['svdcut'] = 'SVD cut $=$'
        
        if major_ticks in ['t0', 'td', 'corr_type', 'fit_type', 'sorting', 'prior_En']:
            major_ticks = 'gevp/'+major_ticks

        fit_args_emph = self.get_fit_args(particles=kwargs['particles'])
        if self.gevp_key in kwargs['particles']:
            rep_particle = self.gevp_key
        else:
            rep_particle = list(fit_args_emph)[0]

        if kwargs.get('t_start') is None:
            if major_ticks == 't_start':
                if self.auto_fit_args is not None and self.auto_fit_args['enabled']:
                    t_plot_min = self.auto_fit_args['t_start']
                    t_plot_max = self.auto_fit_args['t_end']
                else:
                    t_plot_min = int((4 + fit_args_emph[rep_particle]['t_start'])/3)
                    t_plot_max = fit_args_emph[rep_particle]['t_end'] - 1

                kwargs['t_start'] = range(t_plot_min, t_plot_max)
            else:
                kwargs['t_start'] = fit_args_emph[rep_particle]['t_start']

        if kwargs.get('t_end') is None:
            if major_ticks == 't_end':
                if self.auto_fit_args is not None and self.auto_fit_args['enabled']:
                    t_plot_min = self.auto_fit_args['t_start']
                    t_plot_max = self.auto_fit_args['t_end']
                else:
                    t_period = np.min([self._get_t_period(part) for part in kwargs['particles']])
                    t_plot_min = fit_args_emph[rep_particle]['t_start'] + 1
                    t_plot_max = int((2 *fit_args_emph[rep_particle]['t_end'] + t_period)/3)

                kwargs['t_end'] = range(t_plot_min, t_plot_max)
            else:
                kwargs['t_end'] = [fit_args_emph[rep_particle]['t_end']]

        if kwargs.get('t0') is None and major_ticks == 'gevp/t0':
            t_plot_min = np.max([1, self.gevp_args['t0'] - 5])
            if kwargs.get('td') is None:
                t_plot_max = self.gevp_args['td'] - 1
            else:
                t_plot_max = np.min(kwargs['td']) - 1

            kwargs['t0'] = range(t_plot_min, t_plot_max)

        if kwargs.get('td') is None and major_ticks == 'gevp/td':
            if kwargs.get('t0') is None:
                t_plot_min = self.gevp_args['t0'] + 1
            else:
                t_plot_min = np.max(kwargs['t0']) + 1

            t_plot_max = np.min([self._get_t_period(rep_particle), self.gevp_args['td'] + 5])
            kwargs['td'] = range(t_plot_min, t_plot_max)

        fit_args_list = self.get_fit_args(**kwargs)
        fits = self.get_fits(**kwargs)
        
        if param is None or param == 'E0':
            spectra = self.get_spectrum(interaction_only=interaction_only, **kwargs)
            labels = ['$E_0$ ' +tuple_to_str(l)+'' for l in list(spectra[0])]
            vals_list = np.array([[s[k][0] for k in s] for s in spectra]).transpose()
            best_spectrum = self.get_spectrum(interaction_only=interaction_only, particles=kwargs['particles'])
            best_vals = [best_spectrum[k][0] for k in best_spectrum]

        else:
            labels = [param]
            vals_list = np.array([[
                fit.p[param][0]
                if (param in fit.p and hasattr(fit.p[param], '__len__'))
                else fit.p[param] if param in fit.p
                else None for fit in fits]])
            
            best_vals = None
            best_fit = self.get_fits(particles=kwargs['particles'])
            if param in best_fit.p and hasattr(best_fit.p[param], '__len__'):
                best_vals = [best_fit.p[param][0]]
            elif param in best_fit.p:
                best_vals = [best_fit.p[param]]

        if part_src_snk is not None:
            if not part_src_snk in list(spectra[0]):
                raise ValueError('Invalid choice of part_src_snk')
            
            idx = list(spectra[0]).index(part_src_snk)

            vals_list = [vals_list[idx]]
            labels = [labels[idx]]
            best_vals = [best_vals[idx]]

        ic_per_tick = 'logGBF'
        if np.any([kwargs.get(k) is not None and not isinstance(kwargs.get(k), str) and hasattr(kwargs.get(k), '__len__') and len(kwargs.get(k)) > 1 
                for k in ['svdcut', 't0', 'td', 'corr_type', 'fit_type', 'eig_sorting']]):
            ic_per_tick=None
            show_avg=False

        return self._plot_stability(
            fit_args_list, vals_list, labels, major_ticks, best_vals, fit_args_emphasis=fit_args_emph, 
            particles=kwargs['particles'], xlabel=xlabel, show_all=show_all, show_avg=show_avg, 
            minor_ticks=minor_ticks, ic_per_tick=ic_per_tick, ylim=ylim, debug=debug)



class FitArgs(corrfit.two_pt.FitArgs):
    def __init__(self, data, fit_args, gevp_args, gevp_key):
        
        output = {}
        output.update(fit_args)
        if gevp_key in fit_args:
            output[gevp_key]['gevp'] = {}
            output[gevp_key]['gevp'].update(gevp_args)

        super().__init__(data, output)



class FittersDict(corrfit.two_pt.FittersDict):
    def __init__(self, gevp, gevp_args, prior, prior_ratio, svdcut=None):
        data = {}
        data.update(gevp.averaged_correlators_gevp)
        data.update(gevp.averaged_correlators_passthrough)

        self.gevp = gevp
        self.gevp_args = gevp_args
        self.data = data
        self.prior = prior
        self.prior_ratio = prior_ratio
        self.svdcut = svdcut


    def _make_fitter(self, fit_args):
        gevp_args = {}
        if self.gevp.gevp_key in fit_args:
            gevp_args.update(self.gevp_args)
            gevp_args.update(fit_args[self.gevp.gevp_key]['gevp'])
        
            correlators = self.gevp.optimized_correlators(
                corr_type=gevp_args['corr_type'],
                ratio=bool(gevp_args['fit_type'] == 'ratio'),
                ratio_particles=gevp_args['ratio_den'],
                t0=gevp_args['t0'],
                td=gevp_args['td'],
                max_states=gevp_args['n_states'],
                sorting_algorithm=gevp_args['sorting']
            )

        else:
            correlators = self.gevp.averaged_correlators_passthrough

        correlators = {
            (part, src_snk) : correlators[(part, src_snk)] 
            for part, src_snk in correlators if part in fit_args
        }

        if 'fit_type' in gevp_args and gevp_args['fit_type'] == 'ratio':
            prior = self.prior_ratio
        else:
            prior = self.prior

        return corrfit.fitters.TwoPtFitter(
            data=correlators, 
            prior=prior,  
            fit_args=fit_args,
            svdcut=self.svdcut
        )





