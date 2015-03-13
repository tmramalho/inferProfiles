# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:39:07 2015

@author: Eric

Create plots that compare results from the Bcd and gap-gene datasets. Currently requires scalingBicoidFinalReally and scalingMutantAll to have been run.

Suffixes at the end of variable names (Eric's convention):
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s: string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples
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
    
    tag_below_l = ['Hb_Bcd2x',
                   'ind_213',
                   'Kni_Bcd2X',
                   'Kr_Bcd2X',
                   'scaling_large',
                   'scaling_large(100)']
    # Specify tags to plot on the underside of the point for clarity  
    tag_right_l = ['Gt_bcdE1',
                   'ind_26', 
                   'ind_152', 
                   'Kni_Bcd2X',
                   'Kni_etsl',
                   'LEandSE_dor', 
                   'scaling_large',
                   'scaling_large(100)']
    # Specify tags to plot on the right side for clarity    
    
    plt.figure(figsize=(16, 12))
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
            sigma_l = dataset[-1]['norm_sigma_l']
            col = 'r' if pca_std[1] < 0.05 else 'b'
            plt.errorbar(100*sigma_l, r_sq, yerr=r_ci,marker='o', ms=np.sqrt(s_size),
	                   alpha=0.5, c=col)
            if dataset[5] in tag_below_l:
                y_sign = -1
                va = 'top'
            else:
                y_sign = 1
                va = 'bottom'
            if dataset[5] in tag_right_l:
                x_sign = 1
                ha = 'left'
            else:
                x_sign = -1
                ha = 'right'
            plt.annotate(dataset[5], xy = (100*sigma_l, r_sq),
                         xytext = (x_sign*20, y_sign*20),
				  textcoords = 'offset points', ha = ha, va = va,
				  bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
				  arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', ec='0.1'))
    plt.title('Summary of all data')
    plt.ylabel('R-squared')
    plt.xlabel('sigma_L (% egg length)')
    plt.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                  'r_sq_vs_sigma_l.pdf')))
    
    

if __name__ == '__main__':
    """ Notes on plots:
    - make_summary_plot: we're only passing late-stage embryos to it for clarity.
    """
    
    # Import only selected data
    (pca_bcd_a, __) = np.load(os.path.join(config.tmp_path, 'results.npy'))
    pca_bcd = list(pca_bcd_a)
    (pca_mutant_d, __) = np.load(os.path.join(config.tmp_path, 'mutant_all_res.npy'))
    pca_mutant = []
    mutants = ['etsl', 'Bcd2X', 'bcdE1']
    for i, key in enumerate(pca_mutant_d.keys()):
        mutant_name = scalingMutantAll.trim_name(pca_mutant_d[key][0][5])[4:]
        if mutant_name not in mutants:
            continue
        for dataset in pca_mutant_d[key]:
            gene_name, __, stage = dataset[5].split(' ')
            if stage == 'early':
                continue
            dataset_l = list(dataset)
            dataset_l[5] = gene_name + '_' + mutant_name
            pca_mutant.append(tuple(dataset_l))
    pca_results = pca_bcd + pca_mutant
    
    # Remove ventral datasets
    pca_results = [dataset for i, dataset in enumerate(pca_results)
                   if 'ven' not in dataset[5]]  
    
    # Create plot of R^2 vs. sigma_L for all datasets
    make_summary_plot(pca_results)