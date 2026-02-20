import numpy as np
import gvar as gv

import corrfit.base
import corrfit.two_pt.models

class Fitter(corrfit.base.Fitter):

    def __init__(self, data, prior, fit_args, p0=None, build_prior=True, svdcut=None):
        super().__init__(data, prior, fit_args, p0, build_prior, svdcut=svdcut)


    @property
    def models(self):
        models = np.array([])

        for part, src_snk in self.data:

            if self.fit_args.get((part, src_snk))['perform_fit']:
                t_start = self.fit_args.get((part, src_snk))['t_start']
                t_end = self.fit_args.get((part, src_snk))['t_end']
                n_states = self.fit_args.get((part, src_snk))['n_states']
                overlap = self.fit_args.get((part, src_snk))['overlap']
                use_log_dE = self.fit_args.get((part, src_snk))['use_log_dE']
                t0_offset = self.fit_args.get((part, src_snk))['t0_offset']

                if self.fit_args.get((part, src_snk))['particle_statistics'] == 'fermi-dirac':
                    
                    datatag = ('two_pt', part, src_snk)
                    models = np.append(models,
                            corrfit.two_pt.models.BaryonModel(datatag, t=range(int(t_start), int(t_end)),
                            n_states=n_states, overlap=overlap, use_log_dE=use_log_dE, t0_offset=t0_offset, p_keys=self.p_keys[(part, src_snk)]))

                elif self.fit_args.get((part, src_snk))['particle_statistics'] == 'bose-einstein':
                    raise NotImplementedError
                    datatag = ('two_pt', part, src_snk)
                    models = np.append(models,
                                corrfit.two_pt.models.MesonModel(datatag, t=range(int(t_start), int(t_end)), t_period=self.t_period,
                                n_states=n_states, overlap=overlap, use_log_dE=use_log_dE, p_keys=self.p_keys[(part, src_snk)]))
        
        return models


    @property
    def spectrum(self):
        p = self.fit.p

        output = {}
        for p_ss in self.p_keys:
            n_states = self.fit_args.get(p_ss)['n_states']
            use_log_dE = self.fit_args.get(p_ss)['use_log_dE']

            E = []
            for n in range(n_states+1):
                if 'E'+str(n) in self.p_keys[p_ss]:
                    E.append(p[self.p_keys[p_ss]['E'+str(n)]])

            if len(E) < n_states:
                if use_log_dE:
                    En = np.sum(E)
                    dE = np.exp(p[self.p_keys[p_ss]['log(dE)']])
                    E = E + [En + dE[j] for j in range(n_states-len(E))]
                else:
                    En = np.sum(E)
                    dE = p[self.p_keys[p_ss]['dE']]
                    E = E + [En + dE[j] for j in range(n_states-len(E))]

            output[p_ss] = E

        return output


    def _build_p_keys(self):
        '''returns a nested dict with keys matching those of data, to wit,
        {(part, src_snk) : {param : input_prior_label}}}
        e.g., {('proton', ('S', 'S')) : {'E0' : 'E0::proton::(S,S)'}}}
        '''
        tuple_to_str =  lambda t : str(t) if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'
        #contains_str_or_int = lambda e, l : bool(e in [str(li) for li in l])

        def contains_str_or_int(e, l):
            if not hasattr(l, '__len__') or isinstance(l, str):
                l = [l]

            return bool(str(e) in [str(li) for li in l])

        output = {}
        for label in self._input_prior:        
            if len(label.split('::')) == 3:
                param, part, src_snk = label.split('::')
                if param.startswith('log('):
                    param = param + ')'
                    src_snk = src_snk[:-1]
                    all_src_snks = [s for p, s in self.data if p == part and tuple_to_str(s) == src_snk]
                elif param == 'Z':
                    all_src_snks = [s for p, s in self.data if p == part and contains_str_or_int(src_snk, s)]
                else:
                    all_src_snks = [s for p, s in self.data if p == part and tuple_to_str(s) == str(src_snk)]
            elif len(label.split('::')) == 2:
                param, part = label.split('::')
                if param.startswith('log('):
                    param = param + ')'
                    part = part[:-1]
                all_src_snks = [s for p, s in self.data if p == part]
            else:
                raise ValueError('Malformed prior keys')

            # since these are sorted, labels without a src_snk are overwritten
            # e.g., 'wf::proton' is replaced by 'wf::proton::(S,S)' (if available)
            for src_snk in all_src_snks:
                if (part, src_snk) not in output:
                    output[(part, src_snk)] = {}

                if label.startswith('log(E0') or label.startswith('E0'):
                    output[(part, src_snk)][param] = label
                elif (label.startswith('E') or label.startswith('log(E')) and self.fit_args.get((part, src_snk))['prior_En']:
                    n = int(param[1:])
                    if n < self.fit_args.get((part, src_snk))['n_states']:
                        output[(part, src_snk)][param] = label
                elif label.startswith('log(dE') and self.fit_args.get((part, src_snk))['use_log_dE'] and self.fit_args.get((part, src_snk))['n_states'] > 1:
                    output[(part, src_snk)][param] = label
                elif label.startswith('dE') and not self.fit_args.get((part, src_snk))['use_log_dE'] and self.fit_args.get((part, src_snk))['n_states'] > 1:
                    output[(part, src_snk)][param] = label
                elif label.startswith('wf') and self.fit_args.get((part, src_snk))['overlap'] != 'ZZ':
                    output[(part, src_snk)][param] = label
                elif label.startswith('Z') and self.fit_args.get((part, src_snk))['overlap'] == 'ZZ':
                    if not hasattr(src_snk, '__len__') or isinstance(src_snk, str):
                        safe_ss = [src_snk]
                    else:
                        safe_ss = src_snk
                    if len(safe_ss) == 2:
                        if label.split('::')[-1] == str(safe_ss[0]):
                            output[(part, src_snk)]['Z_src'] = label
                        if label.split('::')[-1] == str(safe_ss[1]):
                            output[(part, src_snk)]['Z_snk'] = label
                    elif len(label.split('::')) == 2 or label.split('::')[-1] == str(safe_ss[0]):
                        output[(part, src_snk)]['Z_src'] = label
                        output[(part, src_snk)]['Z_snk'] = label

        return output
    

    def _build_prior(self, input_prior=None):
        if input_prior is None:
            input_prior = self._input_prior

        def set_prior_E0(input_prior, p_key, p_ss):
            if self.fit_args.get(p_ss)['particle_statistics'] == 'fermi-dirac' or self.fit_args.get(p_ss)['overlap'] == 'ZZ':
                return input_prior[p_key]
            #elif self.fit_args.get(p_ss)['particle_statistics'] == 'bose-einstein' and self.fit_args.get(p_ss)['overlap'] != 'ZZ':
            #    m = gv.mean(input_prior[p_key])
            #    s = gv.sdev(input_prior[p_key])
            #    return np.log(gv.gvar(m, s))

        def set_prior_dE(input_prior, p_key, n, p_ss):
            if n_states > 1:
                if self.fit_args.get(p_ss)['use_log_dE']:
                    m = gv.mean(input_prior['log(dE'+p_key[6:]])
                    s = gv.sdev(input_prior['log(dE'+p_key[6:]])
                    return [gv.gvar(m, s) for ni in range(n)]
                else:
                    m = gv.mean(input_prior[p_key])
                    s = gv.sdev(input_prior[p_key])
                    if self.fit_args.get(p_ss)['energy_gaps'] == 'constant':
                        return [np.log(gv.gvar(m, s)) for ni in range(n)]
                    elif self.fit_args.get(p_ss)['energy_gaps'] == '1/n':
                        return [np.log(gv.gvar(m/(ni+1), s)) for ni in range(n)]
                    elif self.fit_args.get(p_ss)['energy_gaps'] == '1/n^2':
                        return [np.log(gv.gvar(m/(ni+1)**2, s)) for ni in range(n)]

        def set_prior_Z(input_prior, p_key, n_states):
            m = gv.mean(input_prior[p_key])
            sd = gv.sdev(input_prior[p_key])
            return [gv.gvar(m, sd) if n==0 else gv.gvar(m, 3 *m) for n in range(n_states)]

        def set_prior_wf(input_prior, p_key, n_states):
            m = gv.mean(input_prior[p_key])
            sd = gv.sdev(input_prior[p_key])
            return [gv.gvar(m, sd) if n==0 else gv.gvar(m, 2 *sd) for n in range(n_states)]
        
        # get number of directly priored states

        prior = gv.BufferDict()
        for part, src_snk in self.data:
            if self.fit_args.get((part, src_snk))['perform_fit']:
                num_priored_states = len([k for k in self.p_keys[(part, src_snk)] 
                    if k.startswith('E')])
                
                n_states = self.fit_args.get((part, src_snk))['n_states']
                for key in self.p_keys[(part, src_snk)]:
                    p_key = self.p_keys[(part, src_snk)][key]

                    if key.startswith('E'):
                        n = int(key[1:])
                        if n < n_states:
                            prior[p_key] = set_prior_E0(input_prior, p_key, (part, src_snk))
                    elif key == 'dE':
                        if n_states>num_priored_states:
                            prior['log('+p_key+')'] = set_prior_dE(input_prior, p_key, n_states-num_priored_states, (part, src_snk))
                    elif key == 'log(dE)':
                        if n_states>num_priored_states:
                            prior[p_key] = set_prior_dE(input_prior, p_key, n_states-num_priored_states, (part, src_snk))
                    elif key in ['Z_src', 'Z_snk']:
                        prior[p_key] = set_prior_Z(input_prior, p_key, n_states)
                    elif key == 'wf':
                        prior[p_key] = set_prior_wf(input_prior, p_key, n_states)
                    elif key == 'En':
                        prior[p_key] = [g for g in input_prior[p_key]]
                    else:
                        print('Missing?', key, self.p_keys[(part, src_snk)][key])
                        print('^^^^')

        # prettify output/group keys together
        output = gv.BufferDict()
        for key_type in ['E', 'log(E', 'dE', 'log(dE', 'wf', 'Z']:
            for key in sorted(list(prior)):
                if key.startswith(key_type):
                    output[key] = prior[key]

        return output