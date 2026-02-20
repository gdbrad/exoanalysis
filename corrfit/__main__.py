'''
import numpy as np
import gvar as gv
import lsqfit
import sys
import os
import matplotlib.pyplot as plt
import re
import argparse
import pathlib
from tqdm import tqdm

import corrfit

#plt.rcParams['figure.figsize'] = [10, 8]

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
plt.rcParams['font.size']  = 20
plt.rcParams['legend.fontsize'] =  16
plt.rcParams["lines.markersize"] = 5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.usetex'] = True

def main():
    
    project_path = os.getcwd()
    sys.path.append(project_path)

    # read "load_data.py" file near data file
    from load_data import InputOutput

    parser = argparse.ArgumentParser(description='Fit correlators.')
    parser.add_argument(
        '-c', '--collection', metavar='collection', type=str, default=None,
        help='fit with priors, args specified in /results/[collection]/{prior.yaml,fit_args.yaml} and save results there; if unspecified, make effective mass plots')
    parser.add_argument(
        '-d', '--data', metavar='data', type=str, default=None,
        help='Specify data file containing raw correlator data')
    parser.add_argument(
        '-e', '--ensemble', metavar='ens', type=str, nargs='*', #choices=input_output.ensembles,
        help='fit specified ensemble; if unspecified, fit all')
    parser.add_argument(
        '-p', '--particle', metavar='particle', type=str, 
        help='fit specified particle')
    #parser.add_argument(
    #    '-rw', '--reweighting', dest='use_reweightings', default=True, action='store_false',
    #    help='use charm reweighting'
    #)
    parser.add_argument(
        '-nf', '--no-fits', dest='perform_fits', default=True, action='store_false',
        help='make plots without performing fits'
    )
    parser.add_argument(
        '-af', '--auto-fits', dest='auto_fits', default=False, action='store_true',
        help='automatically determine optimal t_min/t_max per arxiv/2008.01069 and arxiv/2003.12130'
    )
    parser.add_argument(
        '-ap', '--auto-priors', dest='auto_priors', default=False, action='store_true',
        help='automatically determine optimal piors (only supported for bose-einstein statistics)'
    )
    parser.add_argument(
        '-bs', '--bootstrap', dest='bootstrap_fits', default=False, action='store_true',
        help='bootstrap fits'
    )


    args = vars(parser.parse_args())
    print('Settings:', args, '\n')

    input_output = InputOutput(project_path=project_path, collection=args['collection'], file_h5=args['data'])
    if args['ensemble'] is None:
        if not args['perform_fits']:
            ensembles = input_output.ensembles
        elif args['perform_fits'] and args['auto_fits'] and args['auto_priors']:
            ensembles = input_output.ensembles
        elif args['perform_fits'] and args['auto_fits']:
            ensembles = [ens for ens in input_output.ensembles 
                        if input_output.get_prior(args['particle'], ens) is not None]
        elif args['perform_fits'] and not args['auto_fits']:
            ensembles = [ens for ens in input_output.ensembles 
                        if input_output.get_fit_settings(args['particle'], ens) is not None]
    else:
        ensembles = args['ensemble']
    ensembles = sorted(list(set(ensembles) & set(input_output.get_ensembles(args['particle']))))

    if not os.path.exists(os.path.join(project_path, 'results', input_output.collection)):
        os.makedirs(os.path.join(project_path, 'results', input_output.collection))


    particle = args['particle']
    particle_statistics = input_output.get_particle_statistics(particle)
    pbar_ens = tqdm(ensembles, position=0)
    for ens in pbar_ens:
        pbar_ens.set_description('Ensemble: %s'%(ens))

        # Automatically determine priors
        
        if args['auto_priors'] and particle_statistics == 'bose-einstein':
            fit_manager = corrfit.fit_manager.FitManager(
                particle=particle, 
                ensemble=ens,
                fit_args=fit_args,
                prior=prior,
                use_highest_weight=args['auto_fit']
            )
            data_loader.save_prior(fit_manager.determine_prior(ens))
        else:
            prior = data_loader.prior

        # Save figs without fits
        if not args['perform_fits']:

            fit_manager = corrfit.fit_manager.FitManager(
                particle=particle, 
                ensemble=ens,
                input_output=input_output,
                fit_settings=None,
                use_highest_weight=False
            )

            fig = fit_manager.plot_effective_wf()
            input_output.save_fig(fig, 'figs/%s/%s_eff_wf'%(particle, ens))

            fig = fit_manager.plot_effective_mass()
            input_output.save_fig(fig, 'figs/%s/%s_eff_mass'%(particle, ens))

            str_output = '![image](./figs/%s/%s_eff_wf.png)\n'%(particle, ens) 
            str_output += '![image](./figs/%s/%s_eff_mass.png)\n'%(particle, ens)
            input_output.save_markdown(particle+'-'+ens, str_output)

        # Save figs with fits
        elif args['perform_fits']:
            use_highest_weight = bool(args['auto_fits'])
            #prior = input_output.get_prior(particle=particle, ensemble=ens)

            if use_highest_weight:
                fit_args = None
            else:
                fit_settings = input_output.get_fit_settings(particle=particle, ensemble=ens)

            fit_manager = corrfit.fit_manager.FitManager(
                particle=particle, 
                ensemble=ens,
                input_output=input_output,
                fit_settings=fit_settings,
                use_highest_weight=use_highest_weight
            )
            
            # Save fit index
            if use_highest_weight:
                input_output.save_fit_args(particle=particle, ensemble=ens, fit_args=fit_manager.fit_args)

            if particle_statistics == 'fermi-dirac':
                fig = fit_manager.plot_stability(show_avg=use_highest_weight, show_bayes_factors=args['auto_fits'])
                input_output.save_fig(fig, 'figs/%s/%s_stability_start'%(particle, ens))

                fig = fit_manager.plot_stability(vary='t_end', show_bayes_factors=args['auto_fits'])
                input_output.save_fig(fig, 'figs/%s/%s_stability_end'%(particle, ens))
            elif particle_statistics == 'bose-einstein':
                fig = fit_manager.plot_stability(vary='symmetric')
                input_output.save_fig(fig, 'figs/%s/%s_stability'%(particle, ens))

            fig = fit_manager.plot_effective_wf()
            input_output.save_fig(fig, 'figs/%s/%s_eff_wf'%(particle, ens))

            fig = fit_manager.plot_effective_mass()
            input_output.save_fig(fig, 'figs/%s/%s_eff_mass'%(particle, ens))

            str_output = '```yaml\n'+str(fit_manager)+'```\n'
            str_output += '![image](./figs/%s/%s_eff_wf.png)\n'%(particle, ens) 
            str_output += '![image](./figs/%s/%s_eff_mass.png)\n'%(particle, ens) 

            if particle_statistics == 'fermi-dirac':
                str_output += '![image](./figs/%s/%s_stability_start.png)\n'%(particle, ens) 
                str_output += '![image](./figs/%s/%s_stability_end.png)\n'%(particle, ens) 
            elif particle_statistics == 'bose-einstein':
                str_output += '![image](./figs/%s/%s_stability.png)\n'%(particle, ens) 
                
            input_output.save_markdown(particle+'-'+ens, str_output)

        # Perform bootstrap

        if args['bootstrap_fits']:
            fit_args = input_output.get_fit_args(particle=particle, ensemble=ens)
            #prior = input_output.get_prior(particle=particle, ensemble=ens)

            bootstrapper = corrfit.bootstrapper.Bootstrapper(
                particle=particle, 
                ensemble=ens, 
                fit_args=fit_args)

            bs_results = bootstrapper.bootstrap_param()
            input_output.save_bootstrap_results(ensemble=ens, particle=particle, bs_results=bs_results)
            print('\n')
    

    # Save summary
    md_text = ''

    for file in sorted(os.listdir(os.path.join(project_path, 'results', input_output.collection))):
        pattern_aXXmYYY = re.compile(r'a\d{2}m\d{3}[^\.]*')
        regex = re.search(pattern_aXXmYYY, file)
        if file.endswith('.md') and regex is not None and regex.group() in input_output.ensembles:
            md_text += open(os.path.join(project_path, 'results', input_output.collection, file)).read()

    input_output.save_markdown('README', md_text)


if __name__ == '__main__':
    main()
'''