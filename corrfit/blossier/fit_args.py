import corrfit.base
import copy

class FitArgs(corrfit.base.FitArgs):
    def __init__(self, fargs_unfmt, data_keys=None):
        output = copy.deepcopy(fargs_unfmt)
        for part in output:
            output[part].setdefault('default', {})

            output[part]['default'].setdefault('t_start', None)
            output[part]['default'].setdefault('t_end', None)
            output[part]['default'].setdefault('n_states', None)
            output[part]['default'].setdefault('exp_t0', False)
            output[part]['default'].setdefault('constant_only', False)
            output[part]['default'].setdefault('svdcut', None)
            output[part]['default'].setdefault('xi', None)
            output[part]['default'].setdefault('delta', None)
            output[part]['default'].setdefault('lambda', None)
            output[part]['default'].setdefault('perform_fit', None)

        super().__init__(fargs_unfmt=output, data_keys=data_keys)
        for p_ss in self:
            if self[p_ss]['perform_fit'] is None:
                if (any([self[p_ss][key] is None for  key in ['t_start', 't_end', 'n_states', 'xi', 'delta', 'lambda']])
                        or self[p_ss]['t_start'] >= self[p_ss]['t_end']):
                    self[p_ss]['perform_fit'] = False
                elif (p_ss[1] != 'default' and int(p_ss[1]) >= self[p_ss[0], 'default']['n_states']):
                    self[p_ss]['perform_fit'] = False
                else:
                    self[p_ss]['perform_fit'] = True


    def generate_random_fit_args(self, **kwargs):
        if 't_lower_bound' not in kwargs:
            kwargs['t_lower_bound'] = 4
        
        return super().generate_random_fit_args(**kwargs)