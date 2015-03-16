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



def make_summary_plot(res, plot_name, min_sample_size=12, label_flag=True, title_s=''):
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
                   'Hb_LE&SE',
                   'ind_26',
                   'ind_152',
                   'Kni_Bcd2X',
                   'Kni_etsl',
                   'LEandSE_dor',
                   'scaling_large',
                   'scaling_large(100)']
    # Specify tags to plot on the right side for clarity

    plt.figure(figsize=(16, 12))
    for i, case_t in enumerate(res):
        if case_t[4] < min_sample_size:
            continue
        p1 = np.mean(case_t[1])
        p2 = np.mean(case_t[3])
        samples = case_t[2*np.argmin([p1,p2])+1]
        pca_std = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
        r_samples = np.power(case_t[2*np.argmin([p1,p2])], 2)
        r_sq = np.mean(r_samples)
        r_ci = [[r_sq-np.percentile(r_samples, 2.5)],
                [np.percentile(r_samples, 97.5)-r_sq]]
        s_size = case_t[4]
        assert isinstance(case_t[-1], dict)
        # Make sure that the PCA analysis outputted a dictionary with additional stats
        sigma_l = case_t[-1]['norm_sigma_l']
        col = 'r' if pca_std[1] < 0.05 else 'b'
        plt.errorbar(100*sigma_l, r_sq, yerr=r_ci,marker='o', ms=np.sqrt(s_size),
	                   alpha=0.5, c=col)
        if not label_flag:
            continue
        if case_t[5] in tag_below_l:
            y_sign = -1
            va = 'top'
        else:
            y_sign = 1
            va = 'bottom'
        if case_t[5] in tag_right_l:
            x_sign = 1
            ha = 'left'
        else:
            x_sign = -1
            ha = 'right'
        plt.annotate(case_t[5], xy = (100*sigma_l, r_sq),
                     xytext = (x_sign*20, y_sign*20),
				  textcoords = 'offset points', ha = ha, va = va,
				  bbox = dict(boxstyle = 'round,pad=0.5',
                                 fc = 'yellow', alpha = 0.2),
				  arrowprops = dict(arrowstyle = '->',
                                       connectionstyle = 'arc3,rad=0', ec='0.1'))
    plt.title(title_s)
    plt.ylabel('R-squared')
    plt.xlabel('sigma_L (% egg length)')
    plt.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name)))



if __name__ == '__main__':
    """ Notes on plots:
    - make_summary_plot: we're only passing late-stage embryos to it for clarity.
    """

    # Import data
    (pca_bcd_a, __) = np.load(os.path.join(config.tmp_path, 'results.npy'))
    pca_bcd = list(pca_bcd_a)
    (pca_wt_gap_a, __) = np.load(os.path.join(config.tmp_path, 'wt_gap.npy'))
    pca_wt_gap = list(pca_wt_gap_a)
    (pca_mutant_d, __) = np.load(os.path.join(config.tmp_path, 'mutant_all_res.npy'))


    ## Only plot selected datasets (cases)

    # Filter all but select LE&SE gap gene data
    pca_wt_gap_selected = []
    wt_gap_selected_l = ['LEandSE0_Gt_Dorsal_late', 'LEandSE0_Hb_Dorsal_late',
                         'LEandSE0_Kni_Dorsal_late', 'LEandSE0_Kr_Dorsal_late']
    for case_t in pca_wt_gap_a:
        case_name = case_t[5]
        if case_name not in wt_gap_selected_l:
            continue
        __, gene_name, __, __ = case_name.split('_')
        case_l = list(case_t)
        case_l[5] = gene_name + '_LE&SE'
        pca_wt_gap_selected.append(tuple(case_l))

    # Filter all but select mutants
    pca_mutant_selected = []
    mutants = ['etsl', 'Bcd2X', 'bcdE1']
    for i, key in enumerate(pca_mutant_d.keys()):
        mutant_name = scalingMutantAll.trim_name(pca_mutant_d[key][0][5])[4:]
        if mutant_name not in mutants:
            continue
        for case_t in pca_mutant_d[key]:
            gene_name, __, stage = case_t[5].split(' ')
            if stage == 'early':
                continue
            case_l = list(case_t)
            case_l[5] = gene_name + '_' + mutant_name
            pca_mutant_selected.append(tuple(case_l))

    pca_results_selected = pca_bcd + pca_wt_gap_selected + pca_mutant_selected

    # Remove ventral datasets
    pca_results_selected = [case_t for i, case_t in enumerate(pca_results_selected)
                   if 'ven' not in case_t[5]]

    # Create plot of R^2 vs. sigma_L
    make_summary_plot(pca_results_selected,
                      plot_name='r_sq_vs_sigma_l__selected.pdf',
                      title_s='Summary of dorsal/symmetric Bcd and selected LE&SE and mutant gap gene data')


    ## Plot all datasets (cases)

    # Keep all mutants
    pca_mutant_all = []
    for i, key in enumerate(pca_mutant_d.keys()):
        for case_t in pca_mutant_d[key]:
            pca_mutant_all.append(case_t)

    pca_results = pca_bcd + pca_wt_gap + pca_mutant_all

    # Create plot of R^2 vs. sigma_L
    make_summary_plot(pca_results,
                      label_flag=False,
                      plot_name='r_sq_vs_sigma_l__all.pdf',
                      title_s='Summary of all Bcd and LE&SE and mutant gap gene data')