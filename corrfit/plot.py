import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gvar as gv
import cycler
import lsqfitics

from corrfit.utils import pm


# from https://siegal.bio.nyu.edu/color-palette/
colorblind_cmap = matplotlib.colors.ListedColormap([
    '#0072B2', '#D55E00',  '#009E73', '#E69F00', '#56B4E9',  
    '#CC79A7',  '#F0E442',  '#000000', '#899499'])

# other cmap
custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'my_colormap',
    np.vstack(
        [matplotlib.colormaps['cool_r'](np.linspace(0., 1, 128)),
        matplotlib.colormaps['winter_r'](np.linspace(0, 1, 128))]
    )
)

def default_cmap(val):
    if val == 1/2:
        # better value for 1/2
        return custom_cmap(val+0.1)
    else:
        return custom_cmap(val)
    
#default_cmap = colorblind_cmap

# taken from https://stackoverflow.com/a/56253636
def legend_deduplicated(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    return ax.legend(*zip(*unique), **kwargs)


def _swap_inner_outer_keys(table):
    keys = []
    for t1 in table:
        for t2 in table[t1]:
            if t2 not in keys:
                keys.append(t2)
    keys = sorted(keys)

    out = {}
    for k in keys:
        out[k] = {}
        for t1 in table:
            if k in table[t1]:
                out[k][t1] = table[t1][k]

    return out

def plot_table(table, xlabel=None, ylabel=None, title=None, show_legend=True, swap_keys=False):
    # takes a like 
    #   table[p1][common_keys1]
    #   table[p2][common_keys2]
    # if swap_keys, then like
    #   table[common_keys1][p1]
    #   table[common_keys2][p2]

    if swap_keys:
        table = _swap_inner_outer_keys(table)

    fig, ax = plt.subplots()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0]*1.5, size[1])


    unique_keys = []
    for k1 in table:
        for k2 in table[k1]:
            if k2 not in unique_keys:
                unique_keys.append(k2)

    for i, k2 in enumerate(unique_keys):
        for j, k1 in enumerate(table):
            if k2 in table[k1]:
                x = i + np.linspace(-0.25, 0.25, len(table))[j]
                ax.errorbar(x, y=gv.mean(table[k1][k2]), yerr=gv.sdev(table[k1][k2]), 
                    color=default_cmap(j/(len(table)+1)), capsize=150/len(unique_keys)/len(table), label=k1)

        ax.axvline(i+0.5, ls='--', alpha=0.1)

    ax.axvline(-0.5, ls='--', alpha=0.1)
    ax.set_xlim(-1, len(unique_keys))
    ax.set_xticks(range(len(unique_keys)), unique_keys, rotation='vertical')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_legend:
        legend_deduplicated(ax, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.close()
    return fig


def get_closest_factorization(n):
    # Returns two integers whose product is close to but greater than n
    sqrt_n = int(np.sqrt(n))
    if sqrt_n == np.sqrt(n):
        return (sqrt_n, sqrt_n)
    
    for j in range(0, sqrt_n + 1):
        #print(j, (sqrt_n + 1) *(sqrt_n - j) )
        if (sqrt_n + 1) *(sqrt_n - j) >= n:
            pass
        else:
            return (sqrt_n + 1, sqrt_n - j + 1)
        

def plot_split_table(table, xlabel=None, ylabel=None, title=None, emph_keys=[], 
        grid_shape=None, show_all=True, swap_keys=False, ylim=None):
    # takes a table like 
    # table[p1][common_keys1]
    # table[p2][common_keys1]
    # table[p3][common_keys1]

    if swap_keys:
        table = _swap_inner_outer_keys(table)
    
    unique_keys = []
    for k1 in table:
        for k2 in table[k1]:
            if k2 not in unique_keys:
                unique_keys.append(k2)

    
    if not show_all:
        temp = []
        for k in unique_keys:
            if np.sum([k in table[p] for p in table]) > 1:

                temp.append(k)
        unique_keys = temp

    if grid_shape is None:
        grid_shape = get_closest_factorization(len(unique_keys))

    fig, axes = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], sharex=True, gridspec_kw={'wspace':0.4, 'hspace':0.2})
    size = fig.get_size_inches()
    fig.set_size_inches(size[0] *grid_shape[1]/2, size[1] *grid_shape[0]/2)

    for j in range(grid_shape[0] * grid_shape[1] - len(unique_keys)):
        fig.delaxes(axes.flatten()[-(j+1)])

    max_j = len(table)
    for k2, ax in zip(unique_keys, axes.ravel()):
        for j, k1 in enumerate(table):
            if k2 in table[k1] and k2 in unique_keys:
                color = default_cmap(j/(max_j+1))

                x = (j+1) / (max_j+1) - 1/2
                y = table[k1][k2]
                ax.errorbar(x, y=gv.mean(y), yerr=gv.sdev(y), marker='o', lw=5, capsize=10, color=color)
                if k1 in emph_keys:
                    ax.axhspan(pm(y, -1), pm(y, 1), alpha=0.1, color=color)

        ax.set_ylabel(k2)
        ax.set_xticks((np.arange(max_j)+1) / (max_j+1) - 1/2, list(table), rotation='vertical')
        ax.set_xlim(-1/2, 1/2)
        ax.set_ylim(ylim)

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(title)

    plt.close()
    return fig

def plot_autocorrelation(data, xlim=None):
    tuple_to_str =  lambda t : t if (isinstance(t, str) or not hasattr(t, '__len__')) else '(' + ','.join([str(s) for s in list(t)]) + ')'

    auto_corr = gv.dataset.autocorr(data)    
    try:
        auto_corr.items()
    except:
        auto_corr = {'d' : auto_corr}

    avg_auto_corr = gv.dataset.avg_data({key : np.array(auto_corr[key]).T for key in auto_corr})

    fig, axes = plt.subplots(nrows=len(auto_corr), sharex=True, gridspec_kw={'wspace':0, 'hspace':0.1})
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1]*(3 + 4*len(data))/7)

    if len(auto_corr) == 1:
        axes = [axes]

    for j, key in enumerate(auto_corr):
        axes[j].set_prop_cycle(cycler.cycler(color=[default_cmap(v) for v in np.linspace(0, 1, 50)]))
        axes[j].plot(auto_corr[key], alpha=0.8)

        pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
        axes[j].fill_between(range(len(avg_auto_corr[key])), pm(avg_auto_corr[key], -1), pm(avg_auto_corr[key], +1), 
            color='indigo', edgecolor='navy', hatch='xx', zorder=10)
        
        try:
            intersect = next(i for i, v in enumerate(pm(avg_auto_corr[key], 1)) if v < np.exp(-2))
            axes[j].axvline(intersect, color='black', lw='2')
            print('Twice autocorrelation length [', key, ']: ', intersect)
            if xlim is None:
                xlim =(0, 3 *intersect)

        except StopIteration:
            pass
            
        axes[j].set_yscale('log')
        axes[j].axhline(np.exp(-1), ls='--', color='black')
        axes[j].axhline(np.exp(-2), ls='--', color='black')

        if xlim is not None:
            axes[j].set_xlim(xlim)
        axes[j].set_ylim(1e-3, 1)

        axes[j].set_ylabel(r'$\Gamma(t)$ %s'%tuple_to_str(key))
    axes[-1].set_xlabel(r'$t_{\rm MC}$')

    plt.close()
    return fig


def plot_cdf(values, weights=None, mu=None, jackknife=False, 
        xlim=None, xlabel='$x$', show_legend=True, show_all=False):

    if isinstance(values[0], gv.GVar):
        if weights is None:
            weights = np.repeat(1/len(values), len(values))
        avg = lsqfitics.calculate_average(values, weights=weights)
        if mu is None:
            mu = gv.mean(avg)
        sigma = gv.sdev(avg)

    else:
        if weights is None:
            if mu is None:
                mu = np.mean(values)
            sigma = np.std(values)

        else:
            if mu is None:
                mu = np.sum([v *w for v, w in zip(values, weights)]) / np.sum(weights)
            sigma = np.sqrt(np.sum([w *(v-mu)**2 for v, w in zip(values, weights)]) / np.sum(weights))

        if jackknife:
            values = np.array([(v - mu) *np.sqrt(len(values)) + mu for v in values])
            sigma = sigma *np.sqrt(len(values))
            

    if show_all:
        x = np.linspace(values.min(), values.max())
    elif xlim is not None:
        x = np.linspace(xlim[0], xlim[1])
    else:
        x = np.linspace(mu - 4 *sigma, mu + 4 *sigma)

    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        np.exp(-0.5 * (1 / sigma * (x - mu))**2))
    y = y.cumsum()
    y /= y[-1]

    fig, ax = plt.subplots()
    ax.plot(x, y, "k--", linewidth=1.5, label="Normal")
    ax.axvline(mu, color='plum')
    ax.axvspan(mu - sigma, mu + sigma, alpha=0.3, color='plum', label='$x = $%s'%(gv.gvar(mu, sigma)))
    ax.ecdf(gv.mean(values), weights=weights, color='cornflowerblue')

    if show_legend:
        ax.legend()

    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$P(X \leq x)$')

    plt.close()
    return fig