# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:39:07 2015

@author: Eric

Create plots that compare results from the Bcd and gap-gene datasets. Currently
requires scalingBicoidFinalReally and scalingMutantAll to have been run.

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
#from matplotlib import patches
import os

import config
reload(config)
import scalingMutantAll
reload(scalingMutantAll)



def make_summary_plot(ax,
                      res,
                      clean_label_d={},
                      clean_label_flag=False,
                      label_flag=True,
                      min_sample_size=12,
                      nudge_down_s_l=[],
                      nudge_up_s_l=[],
                      plot_left_s_l=[],
                      title_s='',
                      type_by_color_flag=False,
                      xlim_t=(),
                      yaxis_s='r-squared',
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
    - If clean_label_flag, plot small labels to the side according to the
        dictionary clean_label_d.
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

    plt.rcParams['font.family'] = 'Arial'
    for i, case_t in enumerate(res):
        if case_t[4] < min_sample_size:
            continue
        p1 = np.mean(case_t[1])
        p2 = np.mean(case_t[3])
        samples = case_t[2*np.argmin([p1,p2])+1]
        pca_std = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
        p = samples[0]
        p_ci = [[p-pca_std[0]],
                [pca_std[1]-p]]
        r_samples = np.power(case_t[2*np.argmin([p1,p2])], 2)
        r_sq = r_samples[0]
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
        if yaxis_s == 'r-squared':
            y = r_sq
            y_ci = r_ci
        elif yaxis_s == 'p-value':
            y = p
            y_ci = p_ci
        ax.errorbar(sigma_l, y, yerr=y_ci, marker='.', ms=np.sqrt(s_size),
	                   alpha=line_alpha, color=col, elinewidth=line_width)
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
        if clean_label_flag:
            if case_t[5] in clean_label_d:
                if clean_label_d[case_t[5]] in plot_left_s_l:
                    xtext_sign = -1
                    horizontalalignment = 'right'
                else:
                    xtext_sign = 1
                    horizontalalignment = 'left'
                if clean_label_d[case_t[5]] in nudge_up_s_l:
                    y_nudge = 3
                elif clean_label_d[case_t[5]] in nudge_down_s_l:
                    y_nudge = -3
                else:
                    y_nudge = 0
                ax.annotate(clean_label_d[case_t[5]], xy = (sigma_l, y),
                            xytext = (xtext_sign*6, -4+y_nudge),
                            textcoords = 'offset points',
                            horizontalalignment=horizontalalignment)
        else:
            ax.annotate(case_t[5], xy = (sigma_l, y),
                         xytext = (x_sign*20, y_sign*20),
    			    textcoords = 'offset points', ha = ha, va = va,
    			    bbox = dict(boxstyle = 'round,pad=0.5',
                                     edgecolor = 'none',
                                     fc = col, alpha = 0.2),
    			    arrowprops = dict(arrowstyle = '->',
                                           connectionstyle = 'arc3,rad=0', ec='0.1'))
    ax.set_title(title_s)
    ax.set_xlabel(r'$\sigma_L / {\langle L \rangle}$')
    if yaxis_s == 'r-squared':
        ax.set_ylabel(r'$R^2$')
    elif yaxis_s == 'p-value':
        ax.axhline(0.05, color='g', linewidth=0.5)
        ax.set_ylabel(r'$p$-value')
        ax.set_yscale('log')
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
    return ax



def make_presentation_plot_v2(pca_bcd_gfp_l, pca_lese_l, pca_mutant_d):


    ## Create figure for panels 1 and 2: Bcd-GFP and WT-length-distribution gap genes

    fig = plt.figure(figsize=(6, 9))
    plot_name = 'PCA_WTEggLengthDistribution'
    width = 0.75
    height = 0.375
    space = 0.11
    letter_position_t = (-0.15, 1.05)
    letter_size = 24

    # Figure 1: Bcd-GFP
    ax = fig.add_axes([(1-width)/2.0, (1+space)/2.0, width, height])
    pca_bcd_gfp_selected_l = [case for case in pca_bcd_gfp_l
                              if (case[5] == 'scaling_large' or 'ind' in case[5])]
    clean_label_d = {'scaling_large': '(all Bcd-GFP data)'}
    make_summary_plot(ax,
                      pca_bcd_gfp_selected_l,
                      clean_label_d=clean_label_d,
                      clean_label_flag=True,
                      title_s='Individual sessions of Bcd-GFP embryos',
                      xlim_t=(0.023, 0.056),
                      yaxis_s='p-value')
    ax.annotate('A', xy=(0,0), xycoords='axes fraction',
                xytext=letter_position_t, textcoords='axes fraction',
                fontsize=letter_size,
                horizontalalignment='left',
                verticalalignment='bottom')

    # Figure 2: WT-length-distribution gap genes
    ax = fig.add_axes([(1-width)/2.0, (1-2*height-space)/2.0, width, height])
    pca_mutant_selected_l = []
    for key in pca_mutant_d:
        if 'Bcd2X' in pca_mutant_d[key][0][5]:
            pca_mutant_selected_l += pca_mutant_d[key]
    clean_label_d = {'Gt data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP early': 'Gt early',
                     'Hb data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP early': 'Hb early',
                     'Kni data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP early': 'Kni early',
                     'Kr data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP early': 'Kr early',
                     'Gt data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP late': 'Gt late',
                     'Hb data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP late': 'Hb late',
                     'Kni data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP late': 'Kni late',
                     'Kr data_dorsal_130310_Kni_Kr_Gt_Hb_Bcd2X_AP late': 'Kr late'}
    make_summary_plot(ax,
                      pca_mutant_selected_l,
                      clean_label_d=clean_label_d,
                      clean_label_flag=True,
                      nudge_down_s_l=['Gt early'],
                      nudge_up_s_l=['Hb early', 'Hb late'],
                      plot_left_s_l=['Kni late', 'Kni early'],
                      title_s='Staining for gap genes in wild-type embryos',
                      xlim_t=(0.023, 0.056),
                      yaxis_s='p-value')
    ax.annotate('B', xy=(0,0), xycoords='axes fraction',
                xytext=letter_position_t, textcoords='axes fraction',
                fontsize=letter_size,
                horizontalalignment='left',
                verticalalignment='bottom')


    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name + '.pdf')))


    ## Create figure for panel 3: LE&SE Bcd and gap genes

    fig = plt.figure(figsize=(8, 6))
    plot_name = 'PCA_EnlargedEggLengthDistribution'

    # Figure 3
    ax = fig.add_subplot(1, 1, 1)
    pca_bcd_gfp_selected_l = [case for case in pca_bcd_gfp_l
                              if case[5] == 'Temp_varied']
    gene_name_l = ['LEandSE1_Bcd', 'LEandSE0_Gt',
                   'LEandSE0_Hb', 'LEandSE0_Kr',
                   'LEandSE0_Kni']
    pca_lese_selected_l = [case for case in pca_lese_l
                           if np.any([gene_name in case[5] for gene_name in gene_name_l])]
    pca_all_l = pca_bcd_gfp_selected_l + pca_lese_selected_l
    clean_label_d = {'Temp_varied': '(Bcd-GFP, temperature-varied)',
                     'LEandSE1_Bcd_Dorsal_early': 'Bcd early',
                     'LEandSE0_Gt_Dorsal_early': 'Gt early',
                     'LEandSE0_Hb_Dorsal_early': 'Hb early',
                     'LEandSE0_Kni_Dorsal_early': 'Kni early',
                     'LEandSE0_Kr_Dorsal_early': 'Kr early',
                     'LEandSE1_Bcd_Dorsal_late': 'Bcd late',
                     'LEandSE0_Gt_Dorsal_late': 'Gt late',
                     'LEandSE0_Hb_Dorsal_late': 'Hb late',
                     'LEandSE0_Kni_Dorsal_late': 'Kni late',
                     'LEandSE0_Kr_Dorsal_late': 'Kr late',
                     'LEandSE1_Bcd_Dorsal_nc14': 'Bcd nc14',
                     'LEandSE0_Gt_Dorsal_nc14': 'Gt nc14',
                     'LEandSE0_Hb_Dorsal_nc14': 'Hb nc14',
                     'LEandSE0_Kni_Dorsal_nc14': 'Kni nc14',
                     'LEandSE0_Kr_Dorsal_nc14': 'Kr nc14'}
    make_summary_plot(ax,
                      pca_all_l,
                      clean_label_d=clean_label_d,
                      clean_label_flag=True,
                      nudge_down_s_l=['Bcd early', 'Gt early'],
                      nudge_up_s_l=['Kni early'],
                      plot_left_s_l=['Bcd early'],
                      title_s='Staining of LE&SE embryos and temperature-varied embryos',
                      xlim_t=(0.067, 0.085),
                      yaxis_s='p-value')

    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
                                                         plot_name + '.pdf')))



if __name__ == '__main__':
    """ Notes on plots:
    - make_summary_plot: we're only passing late-stage embryos to it for clarity.
    """

    ## Import data

    (pca_bcd_a, __) = np.load(os.path.join(config.tmp_path, 'results.npy'))
    pca_bcd = list(pca_bcd_a)
    (pca_wt_gap_a, __) = np.load(os.path.join(config.tmp_path, 'wt_gap.npy'))
    pca_wt_gap = list(pca_wt_gap_a)
    (pca_mutant_d, __) = np.load(os.path.join(config.tmp_path, 'mutant_all_res.npy'))


#    ## Only plot selected datasets (cases)
#
#    # Of the LE&SE data, select the gap genes and Bcd
#    pca_wt_gap_selected = []
#    wt_gap_selected_l = ['LEandSE1_Bcd_Dorsal_early',
#                         'LEandSE0_Gt_Dorsal_early', 'LEandSE0_Hb_Dorsal_early',
#                         'LEandSE0_Kni_Dorsal_early', 'LEandSE0_Kr_Dorsal_early',
#                         'LEandSE1_Bcd_Dorsal_late',
#                         'LEandSE0_Gt_Dorsal_late', 'LEandSE0_Hb_Dorsal_late',
#                         'LEandSE0_Kni_Dorsal_late', 'LEandSE0_Kr_Dorsal_late',
#                         'LEandSE1_Bcd_Dorsal_nc14',
#                         'LEandSE0_Gt_Dorsal_nc14', 'LEandSE0_Hb_Dorsal_nc14',
#                         'LEandSE0_Kni_Dorsal_nc14', 'LEandSE0_Kr_Dorsal_nc14']
#    for case_t in pca_wt_gap_a:
#        case_name = case_t[5]
#        if case_name not in wt_gap_selected_l:
#            continue
#        __, gene_name, __, stage_s = case_name.split('_')
#        case_l = list(case_t)
#        case_l[5] = 'LE&SE_' + gene_name + '_' + stage_s
#        pca_wt_gap_selected.append(tuple(case_l))
#
#    # Filter out all mutants and keep WT
#    pca_mutant_selected = []
#    mutants = ['Bcd2X']
#    for i, key in enumerate(pca_mutant_d.keys()):
#        mutant_name = scalingMutantAll.trim_name(pca_mutant_d[key][0][5])[4:]
#        if mutant_name not in mutants:
#            continue
#        for case_t in pca_mutant_d[key]:
#            gene_name, __, stage_s = case_t[5].split(' ')
##            if stage_s == 'early':
##                continue
#            case_l = list(case_t)
#            case_l[5] = mutant_name + '_' + gene_name + '_' + stage_s
#            pca_mutant_selected.append(tuple(case_l))
#
#    pca_results_selected = pca_bcd + pca_wt_gap_selected + pca_mutant_selected
#
#    # Remove ventral datasets
#    pca_results_selected = [case_t for i, case_t in enumerate(pca_results_selected)
#                   if 'ven' not in case_t[5]]
#
#    # Create plot of R^2 vs. sigma_L (with labels)
#    fig = plt.figure(figsize=(24, 18))
#    plot_name='r_sq_vs_sigma_l__selected'
#    make_summary_plot(fig,
#                      pca_results_selected,
#                      title_s='Summary of dorsal/symmetric Bcd and gap gene ' +
#                              'data (WT and LE&SE)')
#    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
#                                                         plot_name + '.pdf')))
#
#    # Create plot of p-valueR^2 vs. sigma_L (with labels)
#    fig = plt.figure(figsize=(24, 18))
#    plot_name='p_value_vs_sigma_l__selected'
#    make_summary_plot(fig,
#                      pca_results_selected,
#                      figure_size=(24, 18),
#                      title_s='Summary of dorsal/symmetric Bcd and gap gene ' +
#                              'data (WT and LE&SE)',
#                                  yaxis_s='p-value')
#    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
#                                                         plot_name + '.pdf')))
#
#    # Create plot of R^2 vs. sigma_L (presentation version 1)
#    fig = plt.figure()
#    presentation_kwargs_d = {'label_flag': False,
#                             'title_s': '',
#                             'type_by_color_flag': True,
#                             'xlim_t': (0.02, 0.09),
#                             'ylim_t': (0, 1)}
#    plot_name = 'r_sq_vs_sigma_l__selected__presentation_v1'
#    ax = make_summary_plot(fig,
#                           pca_results_selected,
#                           **presentation_kwargs_d)
#    #ax.add_patch(patches.Rectangle((0.0707, 0), 0.013, 0.8, alpha=0.4, color=(1,1,0)))
#    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
#                                                         plot_name + '.pdf')))
#    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
#                                                         plot_name + '.png')))
#
#
#    ## Plot all datasets (cases)
#
#    # Keep all mutants
#    pca_mutant_all = []
#    for i, key in enumerate(pca_mutant_d.keys()):
#        for case_t in pca_mutant_d[key]:
#            pca_mutant_all.append(case_t)
#
#    pca_results = pca_bcd + pca_wt_gap + pca_mutant_all
#
#    # Create plot of R^2 vs. sigma_L
#    fig = plt.figure()
#    plot_name = 'r_sq_vs_sigma_l__all'
#    make_summary_plot(fig,
#                      pca_results,
#                      label_flag=False,
#                      title_s='Summary of all Bcd and LE&SE and mutant gap gene data')
#    fig.savefig(scalingMutantAll.ensure_dir(os.path.join(config.plots_path, 'summary',
#                                                         plot_name + '.pdf')))


    ## New versions of presentation figures
    make_presentation_plot_v2(pca_bcd, pca_wt_gap, pca_mutant_d)