import corrfit.base

class FitArgs(corrfit.base.FitArgs):
    def __init__(self, fit_args):
        output = {}
        for part in fit_args:
            output[part] = {}
            output[part].setdefault('t_start', None)
            output[part].setdefault('t_end', None)

            if 't_start' in fit_args[part]:
                output[part]['t_start'] = fit_args[part]['t_start']

            if 't_end' in fit_args[part]:
                output[part]['t_end'] = fit_args[part]['t_end']

            if (all([output[part][key] is not None for key in ['t_start', 't_end']])
                    and (output[part]['t_start'] < output[part]['t_end'])):
                output[part]['perform_fit'] = True
            else:
                output[part]['perform_fit'] = False

        super().__init__(output)
