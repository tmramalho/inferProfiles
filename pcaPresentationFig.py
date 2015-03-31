# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:59:54 2015

Plots figure explaning our use of the PCA method (based on plot_pca in scalingBicoidFinalReally.py). Make sure scalingBicoidFinalReally.py has been run, because this script reads in the results from that.

Suffixes at the end of variable names (Eric's convention):
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s (or path): string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples

@author: Eric
"""

from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import scipy.stats as stats
from sklearn.decomposition import PCA

import config
reload(config)
import scalingBicoidFinalReally
reload(scalingBicoidFinalReally)



def add_smart_subplot(num_rows, num_columns, i_plot, width, height,
                      horiz_space, vert_space):
    """
    i_plot indexes which plot this is; horiz_space and vert_space are percentage distance between adjoining figures.
    i_plot, i_row, and i_column are 1-indexed for compatibility with add_subplot function.
    """

    horiz_margin = 0.5 * (1 - num_columns*width - (num_columns-1)*horiz_space)
    vert_margin = 0.5 * (1 - num_rows*height - (num_rows-1)*vert_space)
    assert horiz_margin >= 0 and vert_margin >= 0
    i_column = (i_plot-1) % num_columns + 1
    i_row = int(i_plot-1)/int(num_columns) + 1
    from_left = horiz_margin + (i_column-1)*(width+horiz_space)
    from_bottom = vert_margin + (num_rows-i_row)*(height+vert_space)

    return [from_left, from_bottom, width, height]



def do_pca_analysis(profiles, lens, name=''):
      L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
      profiles_smooth_l = []
      for i,p in enumerate(profiles):
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		average, va = scalingBicoidFinalReally.moving_average(np.log(p+0.001), 46, 100)
		profiles_smooth_l.append(average)
      profiles_a = np.array(profiles_smooth_l)
      pca = PCA(n_components=2)
      pca.fit(profiles_a)
      print pca.explained_variance_ratio_
      profiles_transformed_a = pca.transform(profiles_a)
      m,b,r,p,_ = stats.linregress(L, profiles_transformed_a[:,0])
      p1 = [p]
      r1 = [r]
      for _ in xrange(1000):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[~sample], profiles_transformed_a[~sample,0])
		p1.append(p)
		r1.append(r)
      m,b,r,p,_ = stats.linregress(L, profiles_transformed_a[:,1])
      p2 = [p]
      r2 = [r]
      for _ in xrange(1000):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[~sample], profiles_transformed_a[~sample,1])
		p2.append(p)
		r2.append(r)
      plot_pca(profiles_a, pca, profiles_transformed_a, L, name)
      more_stats_d = {'norm_sigma_l': np.std(lens) / np.mean(lens)}
      return pca, (r1, p1, r2, p2, L.shape[0], name, np.std(L), more_stats_d)



def plot_pca(profiles_a, pca, profiles_transformed_a, L, name):

	# Defining plot position
	fig = plt.figure(figsize=(10.5, 7.5))
	num_columns = 3
	num_rows = 2
	width = 0.22
	height = 0.36
 	horiz_space = 0.10
 	vert_space = 0.15
 	letter_position_t = (-0.15, 1.05)
 	letter_size = 20

 	plt.rcParams['font.family'] = 'Arial'

	lx = np.linspace(np.min(L), np.max(L), 100)
	x = np.linspace(0,1,profiles_a.shape[1])

	# A
	ax = fig.add_axes(add_smart_subplot(num_rows, num_columns, 1, width, height,
                                          horiz_space, vert_space))
	norm = colors.Normalize(vmin=float(np.min(L)), vmax=float(np.max(L)))
	profiles_and_L_a = np.hstack((profiles_a, L.reshape(-1, 1)))
	perm_profiles_and_L_a = np.random.permutation(profiles_and_L_a)
	for profile_and_L_a in perm_profiles_and_L_a:
         m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
         color = m.to_rgba(profile_and_L_a[-1])
         plt.plot(x, np.exp(profile_and_L_a[:-1].T), c=color)
	ax.set_xlabel('AP position (x/L)')
	ax.set_ylabel('Fluorescence (au)')
	plt.title('Raw profiles')
 	ax.annotate('A', xy=(0,0), xycoords='axes fraction',
                  xytext=letter_position_t, textcoords='axes fraction',
                  fontsize=letter_size,
                  horizontalalignment='left',
                  verticalalignment='bottom')

	# B
	ax = fig.add_axes(add_smart_subplot(num_rows, num_columns, 2, width, height,
                                          horiz_space, vert_space))
	simp_profiles_trans1_a = np.vstack((profiles_transformed_a[:, 0],
                                      np.zeros(profiles_transformed_a.shape[0]))).T
	simp_profiles1_a = pca.inverse_transform(simp_profiles_trans1_a)
	norm = colors.Normalize(vmin=float(np.min(L)), vmax=float(np.max(L)))
	simp_profiles1_and_L_a = np.hstack((simp_profiles1_a, L.reshape(-1, 1)))
	perm_simp_profiles1_and_L_a = np.random.permutation(simp_profiles1_and_L_a)
	for profile_and_L_a in perm_simp_profiles1_and_L_a:
         m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
         color = m.to_rgba(profile_and_L_a[-1])
         plt.plot(x, np.exp(profile_and_L_a[:-1].T), c=color)
	ax.set_xlabel('AP position (x/L)')
	ax.set_ylabel('Fluorescence (au)')
	plt.title('Component 1')
	ax.annotate('B', xy=(0,0), xycoords='axes fraction',
              xytext=letter_position_t, textcoords='axes fraction',
              fontsize=letter_size,
              horizontalalignment='left',
              verticalalignment='bottom')

	# C
	ax = fig.add_axes(add_smart_subplot(num_rows, num_columns, 3, width, height,
                                           horiz_space, vert_space))
	simp_profiles_trans2_a = np.vstack((np.zeros(profiles_transformed_a.shape[0]),
                                     profiles_transformed_a[:, 1])).T
	simp_profiles2_a = pca.inverse_transform(simp_profiles_trans2_a)
	norm = colors.Normalize(vmin=float(np.min(L)), vmax=float(np.max(L)))
	simp_profiles2_and_L_a = np.hstack((simp_profiles2_a, L.reshape(-1, 1)))
	perm_simp_profiles2_and_L_a = np.random.permutation(simp_profiles2_and_L_a)
	for profile_and_L_a in perm_simp_profiles2_and_L_a:
         m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
         color = m.to_rgba(profile_and_L_a[-1])
         plt.plot(x, np.exp(profile_and_L_a[:-1].T), c=color)
	ax.set_xlabel('AP position (x/L)')
	ax.set_ylabel('Fluorescence (au)')
	plt.title('Component 2')
	ax.annotate('C', xy=(0,0), xycoords='axes fraction',
              xytext=letter_position_t, textcoords='axes fraction',
              fontsize=letter_size,
              horizontalalignment='left',
              verticalalignment='bottom')

	# D
	ax = fig.add_axes(add_smart_subplot(num_rows, num_columns, 4, width, height,
                                          horiz_space, vert_space))
	plt.scatter(profiles_transformed_a[:,0],
                  profiles_transformed_a[:,1], c=L/float(np.max(L)),
                  cmap=plt.get_cmap('jet'))
	ax.set_xlabel('Component 1')
	ax.set_ylabel('Component 2')
	ax.annotate('D', xy=(0,0), xycoords='axes fraction',
                  xytext=letter_position_t, textcoords='axes fraction',
                  fontsize=letter_size,
                  horizontalalignment='left',
                  verticalalignment='bottom')

	# E
	ax = fig.add_axes(add_smart_subplot(num_rows, num_columns, 5, width, height,
                                          horiz_space, vert_space))
	m,b,r,p,s = stats.linregress(L, profiles_transformed_a[:,0])
	plt.scatter(L, profiles_transformed_a[:,0], c=L/float(np.max(L)),
                  cmap=plt.get_cmap('jet'))
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	ax.set_xlabel(r'$L-\langle L \rangle\ (\mathrm{\mu m})$')
	ax.set_ylabel('Component 1')
 	ax.annotate('E', xy=(0,0), xycoords='axes fraction',
                  xytext=letter_position_t, textcoords='axes fraction',
                  fontsize=letter_size,
                  horizontalalignment='left',
                  verticalalignment='bottom')

	# F
	ax = fig.add_axes(add_smart_subplot(num_rows, num_columns, 6, width, height,
                                          horiz_space, vert_space))
	m,b,r,p,s = stats.linregress(L, profiles_transformed_a[:,1])
	plt.scatter(L, profiles_transformed_a[:,1], c=L/float(np.max(L)),
                  cmap=plt.get_cmap('jet'))
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	ax.set_xlabel(r'$L-\langle L \rangle\ (\mathrm{\mu m})$')
	ax.set_ylabel('Component 2')
 	ax.annotate('F', xy=(0,0), xycoords='axes fraction',
                  xytext=letter_position_t, textcoords='axes fraction',
                  fontsize=letter_size,
                  horizontalalignment='left',
                  verticalalignment='bottom')

	plt.savefig(scalingBicoidFinalReally.ensure_dir(os.path.join(config.plots_path,
        "presentation", "pca_presentation_fig.pdf")))



if __name__ == '__main__':
    (results, __) = np.load(os.path.join(config.tmp_path, 'results.npy'))

    dataset_s = 'scaling_large'
    matching_l = [dataset for dataset in results if dataset[5] == dataset_s]
    assert len(matching_l) == 1
    dataset = matching_l[0]

    #	Load scaling data for all sessions of Bcd-GFP embryos (not temperaturee-varied)
    bcd_sdata = sio.loadmat(os.path.join(config.scaling_data_path,
                                         'ScalingDataLargeSetBcd.mat'),
                            squeeze_me=True)
    bcd_sdata = bcd_sdata['RawData']['M'].item()['Em'].item()
    scaling_lengths = []
    scaling_profiles = []
    for p,l in zip(bcd_sdata[:]['Profile'], bcd_sdata[:]['EL']):
        scaling_lengths.append(l)
        scaling_profiles.append(scalingBicoidFinalReally.smooth_arr_prof(p))
    scaling_profiles = np.array(scaling_profiles)
    scaling_lengths = np.array(scaling_lengths)
    print('{0} embryos'.format(scaling_lengths.shape[0]))

    # Run PCA analysis
    pca, res = do_pca_analysis(scaling_profiles, scaling_lengths, 'scaling_large')