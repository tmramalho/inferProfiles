# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:39:07 2015

@author: Eric

Create plots that compare results from the Bcd and gap-gene datasets. Currently requires scalingBicoidFinalReally and scalingMutantAll to have been run.
"""

import numpy as np
from matplotlib import pyplot as plt
import os

import config
reload(config)
import scalingMutantAll
reload(scalingMutantAll)



def make_summary_plot(res, min_sample_size=11):
    """ From scalingBcdFinalReally.make_summary_plot. 
    - res is a list of results from do_pca_analysis, in scalingBcdFinalReally
            and scalingMutantAll
        - Each entry of res is a tuple containing:
            0 r-value of 1st PC
            1 p-value of 1st PC
            2 r-value of 2nd PC
            3 p-value of 2nd PC
            4 Number of embryos
            5 Name of data set
            6 Standard deviation of embryo lengths
    """
    for i,dataset in enumerate(res):
        if dataset[4] > min_sample_size:
            p1 = np.mean(dataset[1])
            p2 = np.mean(dataset[3])
            samples = dataset[2*np.argmin([p1,p2])+1]
            pca_std = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
            r_samples = np.power(dataset[2*np.argmin([p1,p2])], 2)
            r_sq = np.mean(r_samples)
            r_ci = [[r_sq-np.percentile(r_samples, 2.5)], [np.percentile(r_samples, 97.5)-r_sq]]
            s_size = dataset[4]
            sigma_l = dataset[6]
            col = 'r' if pca_std[1] < 0.05 else 'b'
            plt.errorbar(sigma_l, r_sq, yerr=r_ci,marker='o', ms=np.sqrt(s_size),
	                   alpha=0.5, c=col)
            plt.annotate(dataset[5], xy = (sigma_l, r_sq), xytext = (-20, 20),
				  textcoords = 'offset points', ha = 'right', va = 'bottom',
				  bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
				  arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', ec='0.1'))
    plt.title('Summary of all data')
    plt.ylabel('R-squared')
    plt.xlabel('sigma_L (%)')
    plt.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                  'r_sq_vs_sigma_l')))
    plt.clf()
    
    

if __name__ == '__main__':
    
    # Import data
    (pca_bcd, __) = np.load(os.path.join(config.tmp_path, 'results.npy'))
    (pca_mutant, __) = np.load(os.path.join(config.tmp_path, 'mutant_all_res.npy'))
    # {{{get the cases you want from pca_mutant into a list}}}
    pca_results = pca_bcd + pca_mutant
    
    # Create plot of R^2 vs. sigma_L for all datasets
    make_summary_plot(pca_results)