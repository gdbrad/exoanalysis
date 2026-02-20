import copy
import corrfit.base

class FitArgs(corrfit.base.FitArgs):
    '''
    FitArgs possible values:
    - n_states: int > 0
    - t_start: int < t_end
    - t_end: int > t_start
    - fold_data: boolean (unused)
    - particle_statistics: ['fermi-dirac']
    - use_log_dE: boolean
    - energy_gaps: ['constant', '1/n', '1/n^2']
    - overlap: ['A', 'ZZ']
    - t_period: int > 0
    - t0_offset: int > 0
    - prior_En: boolean
    '''

    def __init__(self, fargs_unfmt, data_keys=None):
        output = copy.deepcopy(fargs_unfmt)
        for part in output:
            output[part].setdefault('default', {})

            # for key in ['use_log_dE', 'energy_gaps', 'particle_statistics', 'overlap', 'prior_En']:
            #     if key in output[part] and output[part][key] is None:
            #         del(output[part][key])

            output[part]['default'].setdefault('n_states', None)
            output[part]['default'].setdefault('t_start', None)
            output[part]['default'].setdefault('t_end', None)
            output[part]['default'].setdefault('use_log_dE', False)
            output[part]['default'].setdefault('fold_data', False)
            output[part]['default'].setdefault('energy_gaps', 'constant')
            output[part]['default'].setdefault('prior_En', False)
            output[part]['default'].setdefault('particle_statistics', 'fermi-dirac')
            output[part]['default'].setdefault('overlap', 'A')
            output[part]['default'].setdefault('t_period', None)
            output[part]['default'].setdefault('t0_offset', None)
            output[part]['default'].setdefault('perform_fit', None)

        super().__init__(fargs_unfmt=output, data_keys=data_keys)
        for p_ss in self:
            if self[p_ss]['perform_fit'] is None:
                if (any([self[p_ss][key] is None for  key in ['n_states', 't_start', 't_end']])
                        or self[p_ss]['t_start'] >= self[p_ss]['t_end']):
                    self[p_ss]['perform_fit'] = False
                else:
                    self[p_ss]['perform_fit'] = True