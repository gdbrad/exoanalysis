import gvar as gv


def pm(g, k=0):
    return gv.mean(g) + k *gv.sdev(g)


# better for labels, legends, etc
def fmt_tuple_as_str(t):
    if hasattr(t, '__len__') and not isinstance(t, str):
        return ','.join([str(ti) for ti in t])
    else:
        return str(t)
    

def dict_full_paths(d=None, sep=' / '):
    # flatten a nested dict
    for j, k in enumerate(d):
        if isinstance(d[k], dict):
            for subkey in dict_full_paths(d[k]):
                yield fmt_tuple_as_str(k) + sep + str(subkey)
        else:
            yield k


def get_from_full_path(d, path, sep=' / '):
    for k in path.split(sep):
        if k in d:
            d = d.get(k)
        elif int(k) in d:
            d = d.get(int(k))
        else:
            return None
    return d