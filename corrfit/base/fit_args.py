import copy
import abc
import itertools
import functools

from ..utils import fmt_tuple_as_str
from ..base.resample import RNG

class FitArgs(dict):
    def __init__(self, fargs_unfmt, data_keys=None):
        output = {}
        for p in fargs_unfmt:
            default = fargs_unfmt[p]['default']
            default.setdefault('uncorrelated', False)
            default.setdefault('svdcut', None)
            output[(p, 'default')] = default

            for ss in fargs_unfmt[p]:
                if ss != 'default':
                    output[(p, fmt_tuple_as_str(ss))] = fargs_unfmt[p][ss]
                    for k in default:
                        output[(p, fmt_tuple_as_str(ss))].setdefault(k, default[k])

            if data_keys is not None:
                for dp, dss in data_keys:
                    if p == dp and (p, fmt_tuple_as_str(dss)) not in output:
                        output[(p, fmt_tuple_as_str(dss))] = {}
                        for k in default:
                            output[(p, fmt_tuple_as_str(dss))].setdefault(k, default[k])

        super().__init__(output)


    def __str__(self):
        output = ''
        max_len = max([len(k) for p_ss in self for k in self[p_ss] if p_ss[1] == 'default' ])
        for p_ss in self:
            all_default = True
            output += str(p_ss)
            for k in self[p_ss]:
                if p_ss[1] == 'default' or self[p_ss[0], 'default'][k] != self[p_ss][k]:
                    all_default = False
                    output += '\n\t'
                    output += str(k).ljust(max_len+2) +' : '+ str(self[p_ss][k])
            if all_default:
                output += '\n\t'
                output += '(default)'.ljust(max_len+2)
            output += '\n\n'

        return output


    # def _pprint(self, d, depth=0):
    #     output = ''
    #     for k in d:
    #         if depth == 0:
    #             output +='\n'
    #         if isinstance(d[k], dict):
    #             output += '   '*depth+ str(k)+'\n'+ self._pprint(d[k], depth=depth+1)
    #         else:
    #             output += '   '*depth+str(k).ljust(25)+': '+str(d[k])+'\n'

    #     return output
    
    @property
    def particles(self):
        return list(set([p for p, _ in self]))
    

    @property
    def sinks(self):
        if list(set([s for _, s in self])) == ['default']:
            return ['default']
        else:
            return sorted(list(set([s for _, s in self if s != 'default'])))
    
    
    def cartesian_product(self, particles=None, src_snk=None, **kwargs):
        # modify all fit_args matching particles and src_snk; leave remainder alone
            
        if kwargs == {}:
            return [copy.deepcopy(self)]
        
        if 'random_models' in kwargs and kwargs['random_models'] == True:
            del(kwargs['random_models'])
            return self.generate_random_fit_args(**kwargs)
        
        if particles is None:
            particles = self.particles
        if isinstance(particles, str):
            particles = [particles]

        if src_snk is None:
            src_snk = 'default'
        else:
            src_snk = fmt_tuple_as_str(src_snk)

        kwargs = {k : kwargs[k] 
            if (hasattr(kwargs[k], '__len__') and not isinstance(kwargs[k], str))
            else [kwargs[k]] 
            for k in kwargs}
        
        output = []
        for x in itertools.product(*kwargs.values()):
            temp = copy.deepcopy(self)
            for p, ss in self:
                if src_snk == ss and p in particles:
                    temp[p, ss].update(dict(zip(kwargs, x)))
                elif src_snk == 'default' and p in particles:
                    temp[p, ss].update(dict(zip(kwargs, x)))

            output.append(temp)

        return output
    
    @functools.cached_property
    def rng(self):
        return RNG()
    

    def generate_random_fit_args(self, dt=1, dt_start=None, dt_end=None, 
            t_lower_bound=None, t_upper_bound=None, n_models=None, n_models_per_sink=5):
        
        if dt is None:
            dt = 1
        if dt_start is None:
            dt_start = dt
        if dt_end is None:
            dt_end = dt

        n_sinks = len(list(self.sinks))
        if n_models is None:
            n_models = n_models_per_sink *n_sinks
        
        j = 0
        models = {}
        while j < n_models:
            try_again = False

            fargs = copy.deepcopy(self)
            key = []
            for p, ss in itertools.product(self.particles, self.sinks):
                if ss != 'default' or self.sinks == ['default']:
                    ts = self.rng.integers(low=-dt_start, high=+dt_start+1)
                    te = self.rng.integers(low=-dt_end, high=+dt_end+1)
                    fargs[(p, ss)]['t_start'] = ts + fargs.get((p, ss))['t_start']
                    fargs[(p, ss)]['t_end'] = te + fargs.get((p, ss))['t_end']

                    # not valid a window
                    if (fargs[(p, ss)]['t_start'] >= fargs[(p, ss)]['t_end']):
                        try_again = True
                    if (t_lower_bound is not None and fargs[(p, ss)]['t_start'] < t_lower_bound):
                        try_again = True
                    if (t_upper_bound is not None and fargs[(p, ss)]['t_end'] > t_upper_bound):
                        try_again = True

                    key.append((p, ss, ts, te))

            key = tuple(key)

            if key in models or try_again:
                pass
            else:
                j += 1
                models[key] = fargs


        return [f for _, f in models.items()]


    def get(self, part_src_snk=None):
        if part_src_snk is not None:
            particle = part_src_snk[0]
            src_snk = part_src_snk[1]

        src_snk = fmt_tuple_as_str(src_snk)

        if (particle, src_snk) in self:
            return self[particle, src_snk]
        elif (particle, 'default') in self:
            return self[particle, 'default']
        else:
            return None
        

    def get_from_path(self, path, part_src_snk=None):
        #raise NotImplementedError
        # access nested dicts like d['key1/key2']
        if part_src_snk is None:
            d = self
        elif part_src_snk in self:
            d = self.get(part_src_snk)
        else:
            return None
            
        #print(path)
        for k in path.split('/'):
            #print(k)
            if k in d:
                d = d.get(k)
            elif int(k) in d:
                d = d.get(int(k))
            else:
                return None
        return d
    

    def full_paths(self, d=None):
        #raise NotImplementedError
        # convert nested dict to 
        if d is None:
            d = self
        for j, k in enumerate(d):
            if isinstance(d[k], dict):
                for subkey in self.full_paths(d[k]):
                    yield self._fmt_tuple_as_str(k) + '/' + str(subkey)
            else:
                yield k


class ModelAvgArgs(dict):

    def __init__(self, margs_unfmt, data_keys=None):
        output = {}
        for p in margs_unfmt:
            default = margs_unfmt[p]['default']
            output[(p, 'default')] = default

        for ss in margs_unfmt[p]:
            if ss != 'default':
                output[(p, fmt_tuple_as_str(ss))] = margs_unfmt[p][ss]
                for k in default:
                    output[(p, fmt_tuple_as_str(ss))].setdefault(k, default[k])

        if data_keys is not None:
            for dp, dss in data_keys:
                if p == dp and (p, fmt_tuple_as_str(dss)) not in output:
                    output[(p, fmt_tuple_as_str(dss))] = {}
                    for k in default:
                        output[(p, fmt_tuple_as_str(dss))].setdefault(k, default[k])

        super().__init__(output)

    def get(self, particle, src_snk=None):

        src_snk = fmt_tuple_as_str(src_snk)

        if (particle, src_snk) in self:
            return self[particle, src_snk]
        elif (particle, 'default') in self:
            return self[particle, 'default']
        else:
            return None