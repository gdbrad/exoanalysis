import numpy as np
import gvar as gv
import corrfit.base.gevp

class GEVP(corrfit.base.gevp.GEVP):

    def construct_energies_overlaps(self, t_max=None, dt=None, 
            construct_energies=True, construct_overlaps=True, 
            use_experimental_construction=False):
        # Returns (energies, overlaps)
        # if construct_energies/overlaps is False, returns None for the respective arg

        if self.off_diagonal_key is None:
            output_overlaps = False

        def get_t0(ti, dt=None):
            if dt is None:
                return int((ti+1)/2) # ceiling ti/2
            else:
                return ti - dt
            
        if t_max is None:
            t_max = np.nanmax([self.raw_correlators[p_ss].shape[1] 
                for p_ss in self.raw_correlators if p_ss[0] == self.gevp_key])

            if self.off_diagonal_key is not None:
                t_max = np.nanmin([t_max, np.nanmax([self.raw_correlators[p_ss].shape[1] 
                    for p_ss in self.raw_correlators if p_ss[0] == self.off_diagonal_key])])
            t_max = t_max - 4
            
        t_min = 4
        t_max = t_max + t_min
        t = np.arange(t_min, t_max)
        output_energies = []
        output_overlaps = []
        for data_rs in self.resampler.resample(means_only=True):
            corr_gevp = self._dict_to_array({p_ss : data_rs[p_ss] for p_ss in data_rs if p_ss[0] == self.gevp_key})
            if construct_overlaps:
                corr_offdiagonal = np.stack([data_rs[p_ss] for p_ss in data_rs if p_ss[0] == self.off_diagonal_key], axis=-1)

            eff_mass = []
            prefactor = []
            optimized_op = []
            for ti in t:
                e2, v = self.eig(corr_gevp, t0=get_t0(ti, dt), td=ti)

                if construct_energies:
                    e1, _ = self.eig(corr_gevp, t0=get_t0(ti, dt), td=ti-1)
                    eff_mass.append(np.log(e1/e2))

                if construct_overlaps:
                    eig_matrix = v.T
                    prefactor.append(np.einsum('ji,jk,kl -> il', eig_matrix.conj(), corr_gevp[ti, :, :], eig_matrix))
                    optimized_op.append(np.einsum('i,ni -> n', np.conj(corr_offdiagonal[ti, :]), v))

            output_energies.append(eff_mass)

            if construct_overlaps:
                prefactor = np.array(prefactor)
                optimized_op = np.array(optimized_op)

                # potentially better defn? (default: false)
                if use_experimental_construction:
                    num = np.array([self.eig(corr_gevp, t0=get_t0(ti, dt), td=ti-1)[0] for ti in t])
                    den = np.array([self.eig(corr_gevp, t0=get_t0(ti, dt), td=ti)[0] for ti in t])
                    exp = np.stack([(num[:, k] / den[:, k])**((t)/2) for k in range(num.shape[1])], -1)

                    output_overlaps.append(np.einsum('tkk, tk, tk -> kt', 
                        1/np.sqrt(prefactor[:-1, :, :]), (optimized_op[:-1, :]), exp[1:, :]))

                # definition per hep-lat/1006.5816
                else:
                    num = np.array([self.eig(corr_gevp, t0=get_t0(ti, dt), td=get_t0(ti, dt)+1)[0] for ti in t])
                    den = np.array([self.eig(corr_gevp, t0=get_t0(ti, dt), td=get_t0(ti, dt)+2)[0] for ti in t])
                    exp = np.stack([(num[:, k] / den[:, k])**((t)/2) for k in range(num.shape[1])], -1)

                    output_overlaps.append(np.einsum('tkk, tk, tk -> kt', 1/np.sqrt(prefactor), (optimized_op), exp))

        if construct_energies:
            output_energies = np.array(output_energies)
            output_energies = {(self.gevp_key+'_energy', k) : output_energies[:, 1:, k] 
                for k in range(output_energies.shape[2])}
        else:
            output_energies = None

        if construct_overlaps:
            output_overlaps = np.abs(output_overlaps)
            output_overlaps = {(self.off_diagonal_key+'_overlap', k) : output_overlaps[:, k, :] 
                for k in range(output_overlaps.shape[1])}
        else:
            output_overlaps = None
        
        return output_energies, output_overlaps