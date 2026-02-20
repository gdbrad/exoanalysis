import numpy as np
import gvar as gv
import os
import h5py
import yaml
import datetime
import matplotlib.pyplot as plt
import pickle
import pathlib

import corrfit.base.resample
from corrfit.utils import get_from_full_path, dict_full_paths


class InputOutput(object):
    def __init__(self, project_path=None, collection=None):
        if project_path is None:
            project_path = os.getcwd()
        self.project_path = project_path

        # Default values
        if collection is None:
            collection = str(datetime.datetime.now())
            for c in [' ', ':', '.', '-']:
                collection = collection.replace(c, '_')
        self.collection = collection
        

    def bin_data(self, data, bin_size=1):
        if bin_size == 1:
            return data
        
        output = dict()
        for key in data:
            n_bins = int(data[key].shape[0]/bin_size)
            #n_bins = int(np.ceil(data[key].shape[0]/bin_size))

            output[key] = np.zeros((n_bins, data[key].shape[1]), dtype=data[key].dtype)
            for j, v in enumerate(data[key].T):
                output[key][:, j] = v[:bin_size *n_bins].reshape(n_bins, bin_size).mean(axis=1)
                #output[key][:, j] = np.append(v[:bin_size *(n_bins-1)].reshape((n_bins-1), bin_size).mean(axis=1), v[bin_size *(n_bins-1):].mean())

        return output


    def _get_fit_settings(self, datatag, ensemble):
        if ensemble is None:
            raise ValueError('Must specify ensemble!')
        
        if datatag is None:
            raise ValueError('Must specify datatag!')

        try:
            with open(self.project_path+'/results/'+self.collection+'/fit_settings.yaml') as file:
                yaml_file = yaml.safe_load(file)
                return yaml_file[ensemble][datatag]
        except:
            # Use default fit_args if file not available in results/collection
            try: 
                with open(self.project_path+'/fit_settings.yaml') as file:
                    yaml_file = yaml.safe_load(file)
                    return yaml_file[ensemble][datatag]
            except: 
                return None


    def get_prior(self, datatags, ensemble):
        if isinstance(datatags, str) or not hasattr(datatags, '__len__'):
            datatags = [datatags]
            
        output = gv.BufferDict()
        for tag in datatags:
            fit_settings = self._get_fit_settings(tag, ensemble)
            if fit_settings is None:
                print('Warning:', ensemble, '/', tag, 'missing prior.')

            else:
                prior_part = fit_settings['prior']
                for key in prior_part:
                    #output[part+'::'+key] = gv.gvar(prior_part[key])
                    if len(key.split('_')) > 1:
                        label = key.split('_')[0]+'::'+tag+'::'+key.split('_')[1]
                    else:
                        label = key+'::'+tag

                    # fix log priors (key should be surrounded by parantheses)
                    if key.startswith('log'):
                        label = label[:6] + label[7:] + ')'

                    output[label] = gv.gvar(prior_part[key])

        # sort keys
        sorted_output = gv.BufferDict()
        for key_type in ['E', 'log(E', 'dE', 'log(dE', 'wf', 'Z']:
            for key in sorted(list(output)):
                if key.startswith(key_type):
                    sorted_output[key] = output[key]

        # sort any other keys
        for key in sorted(list(output)):
            if key not in ['E', 'log(E', 'dE', 'log(dE', 'wf', 'Z']:
                sorted_output[key] = output[key]

        return sorted_output


    def get_model_avg_args(self, datatag, ensemble):
        fit_settings = self._get_fit_settings(datatag, ensemble)
        if (fit_settings is None) or ('model_avg' not in fit_settings):
            print('Warning:', ensemble, '/', datatag, 'missing settings.')
            return {}
        else:
            return fit_settings['model_avg']


    def get_fit_args(self, datatags, ensemble):
        if isinstance(datatags, str) or not hasattr(datatags, '__len__'):
            datatags = [datatags]

        output = {}
        for tag in datatags:
            fit_settings = self._get_fit_settings(tag, ensemble)
            if fit_settings is None:
                print('Warning:', ensemble, '/', tag, 'missing fit args.')

            else:
                if 'fit_args' in fit_settings:
                    output[tag] = fit_settings['fit_args']
                else:
                    output[tag] = {}

        return output
    

    def get_gevp_args(self, datatag, ensemble):
        if datatag is None or ensemble is None:
            raise ValueError('Must specify datatag & ensemble!')

        output = {}
        fit_settings = self._get_fit_settings(datatag, ensemble)
        if fit_settings is None or 'gevp' not in fit_settings:
            print('Warning:', ensemble, '/', datatag, 'missing gevp args.')
        else:
            output = fit_settings['gevp']

        return output
    

    def pickle_gvar_dict(self, var, filepath=None, path=None, filename=None):
        def full_paths(d):
            # convert nested dict to list containing full paths with values
            for j, k in enumerate(d):
                if isinstance(d[k], dict):
                    for subkey, v in full_paths(d[k]):
                        yield str(k) + '/' + str(subkey), v
                else:
                    yield k, d[k]

        var = dict(full_paths(var))

        if filepath is None:
            if path is None:
                path = self.project_path+'/tmp/'

            if filename is None:
                filename = str(datetime.datetime.now())
                for c in [' ', ':', '.', '-']:
                    filename = filename.replace(c, '_')
                filename = filename+'.p'

            filepath = path+'/'+filename

        output = {
            'mean' : dict(gv.mean(var)),
            'cov' : dict(gv.evalcov(var))
        }

        with open(filepath, 'wb') as file:
            pickle.dump(output, file)

        return output


    def unpickle_gvar_dict(self, filepath):
        with open(filepath, 'rb') as file:
            var = pickle.load(file)

        return gv.gvar(var['mean'], var['cov'])


    def plot_raw_correlators(self, data):
        tuple_to_str =  lambda t : t if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'

        fig, axes = plt.subplots(nrows=len(data), sharex=True, gridspec_kw={'height_ratios': [1]*len(data), 'wspace':0, 'hspace':0.1})
        if len(data) == 1:
            axes = [axes]
            
        for j, part_src_snk in enumerate(data):
            for d in data[part_src_snk]:
                axes[j].plot(d)#, lw=0.1)

            axes[j].set_yscale('log')
            axes[j].set_xlabel('$t/a$')
            axes[j].set_ylabel('$C(t)$ '+tuple_to_str(part_src_snk))
            
        plt.close()
        return fig


    def save_fit_args(self, particle, ensemble, fit_args):
        '''Doesn't work -- not likely to be implemented'''
        return None
        try:
            with open(self.project_path+'/results/'+self.collection+'/fit_args.yaml') as file:
                output = yaml.safe_load(file)
        except FileNotFoundError:
            output = {}

        output.update({ensemble : {particle : fit_args}})

        with open(self.project_path+'/results/'+self.collection+'/fit_args.yaml', 'w') as file:
            yaml.dump(output, file, default_flow_style=False, sort_keys=False)

        return None


    def save_markdown(self, title, md_text):
        filename = self.project_path +'/results/'+ self.collection + '/'+ title +'.md'
        pathlib.Path(self.project_path +'/results/'+ self.collection).mkdir(parents=True, exist_ok=True)

        with open(filename, 'w') as file:
            file_content = '# %s\n%s\n'%(title, md_text)
            file.write(file_content)
 
        return None


    def save_to_h5(self, input_dict, sys_err=None, summary=None, filepath=None):
        def full_paths(d):
            # convert nested dict to list containing full paths with values
            for j, k in enumerate(d):
                if isinstance(d[k], dict):
                    for subkey, v in full_paths(d[k]):
                        yield str(k) + '/' + str(subkey), v
                else:
                    yield k, d[k]

        if filepath is None:
            if not os.path.exists(os.path.normpath(self.project_path+'/tmp/')):
                os.makedirs(os.path.normpath(self.project_path+'/tmp/'))

            current_time = str(datetime.datetime.now())
            for c in [' ', ':', '.', '-']:
                current_time = current_time.replace(c, '_')
            filepath = os.path.normpath(self.project_path+'/tmp/'+current_time+'.h5')
        else:
            filepath = os.path.normpath(filepath)

        with h5py.File(filepath, 'a') as h5:

            for path, vals in full_paths(input_dict):
                attrs = {}
                if path in h5:
                    attrs = dict(h5[path].attrs.items())
                    del(h5[path])

                h5.create_dataset(path, data=vals)
                for k, v in attrs.items():
                    h5[path].attrs.create(k, v)

            if sys_err is not None:
                for path, vals in full_paths(sys_err):
                    h5[path].attrs.create('sys_err', vals)

            if summary is not None:
                if 'summary' in h5:
                    del(h5['summary'])
                h5['summary'] = str(summary)

        return None


    def save_fig(self, fig, filename=None, transparent=True, filetype='svg', title=None, **kwargs):
        if filename is None:
            if not os.path.exists(os.path.normpath(self.project_path+'/tmp/')):
                os.makedirs(os.path.normpath(self.project_path+'/tmp/'))

            current_time = str(datetime.datetime.now())
            for c in [' ', ':', '.', '-']:
                current_time = current_time.replace(c, '_')
            output_file = os.path.normpath(self.project_path+'/tmp/'+current_time+'.'+filetype)
        else:
            if self.collection is None:
                raise ValueError('Must specify "collection"!')
            folder_path = os.path.normpath(self.project_path+'/results/'+self.collection+'/figs/')
            pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
            output_file = folder_path+os.path.normpath('/'+filename+'.'+filetype)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        if title is not None:
            fig.suptitle(title)

        fig.savefig(output_file, bbox_inches='tight', transparent=transparent, **kwargs)

        return output_file
    

def bin_data(arr, bin_size):
    output_shape = [int(arr.shape[0] / bin_size)]  + list(arr.shape[1:])

    output = np.zeros(output_shape, dtype=arr.dtype)
    for k in range(int(arr.shape[0] / bin_size)):
        output[k] = np.sum(arr[bin_size*k:(k+1) *bin_size], axis=0) / bin_size

    return output


def fold_data(arr, axis=1):
    T = arr.shape[axis]
    if T % 2 == 1:
        raise ValueError('temporal extent should be even')

    folded_avg = (np.array(arr) + np.flip(arr, axis=axis))/2
    return np.take(folded_avg, range(int(T/2)), axis=axis)


def _add_sys_err(g, sys_err, flatten, sep):
    flatten = True
    return_array = False
    sep = ' / '

    if flatten and not return_array:
        sys_err = {k : get_from_full_path(sys_err, k, sep=sep) 
            for k in dict_full_paths(sys_err, sep=sep)}
        
    for k in sys_err:
        if k in g:
            if hasattr(sys_err[k], '__len__'):
                g_sys = gv.gvar(np.repeat(0, len(sys_err[k])), sys_err[k])
            else:
                g_sys = gv.gvar(0, sys_err[k])
            g[k] = g[k] + g_sys
    return g


def to_gvar(data, rw_factors=None, sys_err=None, svdcut=None, bootstrap=False, jackknife=False, seed=None, n_copies=None,
        preprocessed=False, decorrelate_keys=True, bin_size=1, fold=False, axis_T=1, flatten=False, sep=' / '):
    '''
    Converts either an array or dictionary of measurements into a gvar.
    Supported layouts are either 
        {particle, src_snk) : (cfgs, [...])} 
    (i.e., raw correlator data) or preprocessed bootstrap/jackknife samples 
        {particle, src_snk) : (n_samples+1, [...])}, 
    where sample 0 is the point estimate from the full (non-resampled) dataset

    Args:
        data: measurements
        rw_factors: reweighting measurements with layout (cfgs,)
        preprocessed: specify whether measurements are bootstrap/jackknife samples or raw correlators
        jackknife: either generate covariance matrix using jackknife (preprocessed=False)
            or indicates that samples were generated via jackknife (preprocessed=True)
        bootstrap: either generate covariance matrix using bootstrap (preprocessed=False)
            or indicates that samples were generated via bootstrap (preprocessed=True)
        n_copies: number of bootstrap samples
        seed: bootstrap seed
        decorrelate_keys: return a block-diagonal covariance matrix, decorrelating correlators
            corresponding to different particles/src_snks. Only implemented when either 
            bootstrap=True or jackknife=True.
        svdcut: SVD cut to apply. Only implemented when bootstrap=False and jackknife=False.
        fold: fold data about T/2
        axis_T: axis of temporal extent (when folding)
    '''

    return_array = False
    if not isinstance(data, dict):
        data = {'output' : data}
        return_array = True

    if preprocessed and not jackknife:
        bootstrap = True


    # if flatten is None and return_array is False:
    #     flatten = False
    #     for k in data:
    #         if isinstance(data[k], dict):
    #             flatten = True

    if flatten and not return_array:
        data = {k : get_from_full_path(data, k, sep=sep) 
            for k in dict_full_paths(data, sep=sep)}

    if fold:
        data = {k : fold_data(d, axis=axis_T) for k, d in data.items()}

    if bin_size != 1:
        data = {k : bin_data(d, bin_size=bin_size) for k, d in data.items()}
        if rw_factors is not None:
            rw_factors = bin_data(rw_factors, bin_size=bin_size)

    if bootstrap or jackknife:
        if preprocessed:
            output = {p_ss : data[p_ss] for p_ss in data }
        else:
            # make n_copies bootstrap copies of correlator, only keeping means
            if bootstrap:
                resampler = corrfit.base.resample.BootstrappedData(data, seed=seed, n_copies=n_copies, rw_factors=rw_factors)
            elif jackknife:
                resampler = corrfit.base.resample.JackknifedData(data=data, rw_factors=rw_factors)

            output = {}
            for p_ss in data:
                output[p_ss] = np.zeros([resampler.n_copies + 1] + list(data[p_ss].shape[1:]))

            for j, d_copy in enumerate(resampler.resample(means_only=True)):
                for p_ss in d_copy:
                    output[p_ss][j] = d_copy[p_ss]

        # covert to gvar
        for p_ss in output:
            central = output[p_ss][0]
            mean_rs = np.mean(output[p_ss][1:], axis=0)
            
            # correct for resampling biases
            if bootstrap:
                output[p_ss] = output[p_ss][1:] - mean_rs + central
            elif jackknife:
                output[p_ss] = (output[p_ss][1:] - mean_rs) *np.sqrt(len(output[p_ss][1:])-1) + mean_rs

        if decorrelate_keys:
            output = {p_ss : gv.dataset.avg_data(output[p_ss], spread=True) for p_ss in output}
        else:
            output = gv.dataset.avg_data(output, spread=True)
    
    else:
        if rw_factors is None:
            output = gv.dataset.avg_data(data)

        else:
            temp = {}
            for part_src_snk in data:
                temp[part_src_snk] = np.einsum('ij,i->ij', data[part_src_snk], rw_factors)
            temp['rw'] = rw_factors
            temp = gv.dataset.avg_data(temp)

            output = {p_ss : temp[p_ss] / temp['rw'] for p_ss in data}
    
        if svdcut is None:
            pass
        else:
            output = gv.regulate(output, svdcut=svdcut)

    if sys_err is not None:
        if return_array:
            sys_err = {'output' : sys_err}
        output = _add_sys_err(output, sys_err=sys_err, flatten=flatten, sep=sep)

    if return_array:
        return output['output']
    else:
        return output
        

def h5_tree(file_h5):
    with h5py.File(file_h5, 'r') as f:
        table = []
        f.visititems(lambda name, node: table.append((name, np.shape(node))) if isinstance(node, h5py.Dataset) else None)
    max_len = np.max([len(n[0]) for n in table])

    for path, shape in table:
        print(path.ljust(max_len+2), ':', shape)
