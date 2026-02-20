import numpy as np
import gvar as gv
import warnings
import corrfit.base.gevp
import corrfit.io

from corrfit.plot import plot_autocorrelation

class GEVP(corrfit.base.gevp.GEVP):

    def get_principal_correlators(self, t0=None, max_states=None):
        if t0 is None:
            t0 = self.t0
        if t0 is None:
            raise ValueError("Must specify pivot t0")
        if max_states is None:
            max_states = self.max_states
        
        output = []
        for data_rs in self.resampler.resample(means_only=True):
            corr_gevp = self._dict_to_array({p_ss : data_rs[p_ss] for p_ss in data_rs if p_ss[0] == self.gevp_key})

            principal_corrs_rs = []
            for ti in range(corr_gevp.shape[0]):
                eigval, _ = self.eig(corr_gevp, t0=t0, td=ti)
                principal_corrs_rs.append(eigval)

            output.append(principal_corrs_rs)

        output = np.array(output)
        return {(self.gevp_key+'_prcpl', j) : output[:,:,j] for j in range(output.shape[2]) if j < max_states}


    def get_rotated_correlators(self, t0=None, td=None, max_states=None):
        if t0 is None:
            t0 = self.t0
        if td is None:
            td = self.td
        if max_states is None:
            max_states = self.max_states

        if td == -1:
            td = 2 *t0

        if self.off_diagonal_key is not None:
            smearings_od = [p_ss[1] for p_ss in self.raw_correlators if p_ss[0] == self.off_diagonal_key]

        output_gevp = []
        output_offdiagonal = []
        output_passthrough = {}
        for data_rs in self.resampler.resample(means_only=True):
            corr_gevp = self._dict_to_array({p_ss : data_rs[p_ss] for p_ss in data_rs if p_ss[0] == self.gevp_key})

            _, eigenvectors = self.eig(corr_gevp, t0=t0, td=td)
            eig_matrix = eigenvectors.T

            temp_gevp = np.einsum('ji,tjk,kl -> til', eig_matrix.conj(), corr_gevp, eig_matrix)
            output_gevp.append(temp_gevp)

            if self.off_diagonal_key is not None:
                corr_offdiagonal = np.stack([data_rs[p_ss] for p_ss in data_rs if p_ss[0] == self.off_diagonal_key], axis=-1)
                temp_od = np.einsum('ti,in -> tn', np.conj(corr_offdiagonal), eig_matrix)
                output_offdiagonal.append(temp_od)

            for p_ss in data_rs:
                if p_ss[0] != self.gevp_key:
                    if self.off_diagonal_key is None or p_ss[0] != self.off_diagonal_key:
                        if p_ss not in output_passthrough:
                            output_passthrough[p_ss] = []
                        else:
                            output_passthrough[p_ss].append(data_rs[p_ss])

        output_gevp = np.array(output_gevp)
        output_offdiagonal = np.array(output_offdiagonal)

        output = {}
        output.update({(self.gevp_key+'_rot', j) : output_gevp[:,:,j,j]  
            for j in range(self.max_states) if j < max_states})
        
        if self.off_diagonal_key is not None:
            output.update({(self.off_diagonal_key, (smearings_od[j], str(j))) : output_offdiagonal[:,:,j]  
                for j in range(self.max_states) if j < max_states})
            
        for p_ss in output_passthrough:
            output[p_ss] = np.array(output_passthrough[p_ss])

        return output


    def plot_autocorrelation(self, t0, td, max_states=None):
        if self.bin_size is not None or self.bin_size > 1:
            print("Warning: data already binned (bin_size = %s)"%(self.bin_size))
        # plot autocorrelation of the rotated correlators,
        # freezing eigenvectors of GEVP

        #return self.raw_correlators[list(self.raw_correlators)[0]].shape

        if max_states is None:
            max_states = self.max_states

        corr_gevp = self._dict_to_array({p_ss : self.raw_correlators[p_ss]
            for p_ss in self.raw_correlators if p_ss[0] == self.gevp_key})

        _, eigenvectors = self.eig(np.mean(corr_gevp, axis=0), t0=t0, td=td)
        eig_matrix = eigenvectors.T

        rot_corr = eig_matrix.conj() @ corr_gevp @ eig_matrix
        rot_corr = {(self.gevp_key+'_rot', j) : rot_corr[:,:,j,j]  
                for j in range(self.max_states) if j < max_states}


        return plot_autocorrelation(rot_corr)