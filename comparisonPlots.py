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
from matplotlib import patches
import os

import config
reload(config)
import scalingMutantAll
reload(scalingMutantAll)



def make_summary_plot(res,
                      label_flag=True,
                      min_sample_size=12,
                      title_s='',
                      type_by_color_flag=False,
                      xlim_t=(),
                      ylim_t=()):
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

    line_alpha = 0.5 if label_flag else 1
    line_width = None if label_flag else 2

    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Arial'
    ax = plt.subplot(1, 1, 1)
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
        if type_by_color_flag:
            color_l = [(0.75, 0, 0), (0, 0, 0.5), (1, 0.75, 0.75), (0.75, 0.75, 1)]
            gap_gene_s_l = ['Gt', 'Hb', 'Kr', 'Kni']
            if any([True for s in gap_gene_s_l if s in case_t[5]]):
                col = color_l[2] if pca_std[1] < 0.05 else color_l[3]
            else:
                col = color_l[0] if pca_std[1] < 0.05 else color_l[1]
        else:
            col = 'r' if pca_std[1] < 0.05 else 'b'
        ax.errorbar(sigma_l, r_sq, yerr=r_ci, marker='o', ms=np.sqrt(s_size),
	                   alpha=line_alpha, c=col, elinewidth=line_width)
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
        ax.annotate(case_t[5], xy = (sigma_l, r_sq),
                     xytext = (x_sign*20, y_sign*20),
			    textcoords = 'offset points', ha = ha, va = va,
			    bbox = dict(boxstyle = 'round,pad=0.5',
                                 fc = 'yellow', alpha = 0.2),
			    arrowprops = dict(arrowstyle = '->',
                                       connectionstyle = 'arc3,rad=0', ec='0.1'))
    ax.set_title(title_s)
    ax.set_ylabel(r'$R^2$')
    ax.set_xlabel(r'$\sigma_L / {\langle L \rangle}$')
    if type_by_color_flag:
        legend_handle_l = []
        legend_s_l = ['Bcd (rejection)', 'Bcd (no rejection)',
                      'Gap genes (rejection)', 'Gap genes (no rejection)']
        for i, entry_s in enumerate(legend_s_l):
            handle = plt.errorbar(-1, -1, 0,
                                  alpha=line_alpha,
                                  color=color_l[i],
                                  elinewidth=line_width,
                                  marker='o')
            legend_handle_l.append(handle)
        plt.legend(legend_handle_l, legend_s_l,
                   fontsize=10,
                   frameon=False)

    if xlim_t:
        ax.set_xlim(xlim_t)
    if ylim_t:
        ax.set_ylim(ylim_t)
    return fig, ax



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
    wt_gap_selected_l = ['LEandSE0_Gt_Dorsal_early', 'LEandSE0_Hb_Dorsal_early',
                         'LEandSE0_Kni_Dorsal_early', 'LEandSE0_Kr_Dorsal_early',
                         'LEandSE0_Gt_Dorsal_late', 'LEandSE0_Hb_Dorsal_late',
                         'LEandSE0_Kni_Dorsal_late', 'LEandSE0_Kr_Dorsal_late']
    for case_t in pca_wt_gap_a:
        case_name = case_t[5]
        if case_name not in wt_gap_selected_l:
            continue
        __, gene_name, __, stage_s = case_name.split('_')
        case_l = list(case_t)
        case_l[5] = 'LE&SE_' + gene_name + '_' + stage_s
        pca_wt_gap_selected.append(tuple(case_l))

    # Filter all but select mutants
    pca_mutant_selected = []
    mutants = ['bcdE1']
    for i, key in enumerate(pca_mutant_d.keys()):
        mutant_name = scalingMutantAll.trim_name(pca_mutant_d[key][0][5])[4:]
        if mutant_name not in mutants:
            continue
        for case_t in pca_mutant_d[key]:
            gene_name, __, stage_s = case_t[5].split(' ')
#            if stage_s == 'early':
#                continue
            case_l = list(case_t)
            case_l[5] = mutant_name + '_' + gene_name + '_' + stage_s
            pca_mutant_selected.append(tuple(case_l))

    pca_results_selected = pca_bcd + pca_wt_gap_selected + pca_mutant_selected

    # Remove ventral datasets
    pca_results_selected = [case_t for i, case_t in enumerate(pca_results_selected)
                   if 'ven' not in case_t[5]]

    # Create plot of R^2 vs. sigma_L (with labels)
    plot_name='r_sq_vs_sigma_l__selected'
    (fig, __) = make_summary_plot(pca_results_selected,
                                  title_s='Summary of dorsal/symmetric Bcd and selected LE&SE and mutant gap gene data')
    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name + '.pdf')))

    # Create plot of R^2 vs. sigma_L (presentation version)
    presentation_kwargs_d = {'label_flag': False,
                             'title_s': '',
                             'type_by_color_flag': True,
                             'xlim_t': (0.02, 0.09),
                             'ylim_t': (0, 1)}
    plot_name = 'r_sq_vs_sigma_l__selected__presentation'
    (fig, ax) = make_summary_plot(pca_results_selected,
                                  **presentation_kwargs_d)
    ax.add_patch(patches.Rectangle((0.0707, 0), 0.013, 0.8, alpha=0.4, color=(1,1,0)))
    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name + '.pdf')))
    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name + '.png')))


    ## Plot all datasets (cases)

    # Keep all mutants
    pca_mutant_all = []
    for i, key in enumerate(pca_mutant_d.keys()):
        for case_t in pca_mutant_d[key]:
            pca_mutant_all.append(case_t)

    pca_results = pca_bcd + pca_wt_gap + pca_mutant_all

    # Create plot of R^2 vs. sigma_L
    plot_name = 'r_sq_vs_sigma_l__all'
    (fig, __) = make_summary_plot(pca_results,
                                  label_flag=False,
                                  title_s='Summary of all Bcd and LE&SE and mutant gap gene data')
    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name + '.pdf')))