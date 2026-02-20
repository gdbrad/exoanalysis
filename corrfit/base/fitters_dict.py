import abc

class FittersDict(dict, metaclass=abc.ABCMeta):
    def __init__(self, data, prior, svdcut=None):
        self.data = data
        self.prior = prior
        self.svdcut = svdcut

    
    @abc.abstractmethod
    def _make_fitters(self, fit_args, **kwargs):
        pass

    
    def __repr__(self):
        return str(list(self))
    

    def __str__(self):
        output = ''
        for key in list(self):
            output += '####' *15 + '\n'
            output += str(self._unformat_key(key)) + '\n'
            output += str(self.__getitem__(key).fit) + '\n'

        return output


    def __getitem__(self, fit_args):
        if isinstance(fit_args, dict):
            key = self._format_key(fit_args)
            if key not in self:
                super().__setitem__(key, self._make_fitter(fit_args))
            return super().__getitem__(key)
        elif fit_args in self:
            key = fit_args
            return super().__getitem__(key)
        else:
            return None


    def _format_key(self, fit_args):
        def dict_to_tuple(d):
            return tuple((k, dict_to_tuple(d[k])) 
                  if isinstance(d[k], dict) 
                  else (k, list_to_tuple(d[k])) if isinstance(d[k], list)
                  else (k, d[k]) for k in sorted(d))
        
        def list_to_tuple(l):
            return tuple(list_to_tuple(i) if isinstance(i, list) else i for i in l)

        return dict_to_tuple(fit_args)
    

    def _unformat_key(self, key):
        return self.__getitem__(key).fit_args