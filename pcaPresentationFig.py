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



def do_pca_analysis(profiles, lens, name=''):
      L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
      pr = []
      for i,p in enumerate(profiles):
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		av, va = scalingBicoidFinalReally.moving_average(np.log(p+0.001), 46, 100)
		pr.append(av)
      y = np.array(pr)
      pca = PCA(n_components=2)
      pca.fit(y)
      print pca.explained_variance_ratio_
      yp = pca.transform(y)
      m,b,r,p,_ = stats.linregress(L, yp[:,0])
      p1 = [p]
      r1 = [r]
      for _ in xrange(1000):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[~sample], yp[~sample,0])
		p1.append(p)
		r1.append(r)
      m,b,r,p,_ = stats.linregress(L, yp[:,1])
      p2 = [p]
      r2 = [r]
      for _ in xrange(1000):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[~sample], yp[~sample,1])
		p2.append(p)
		r2.append(r)
      plot_pca(y, pca, yp, L, name)
      more_stats_d = {'norm_sigma_l': np.std(lens) / np.mean(lens)}
      return pca, (r1, p1, r2, p2, L.shape[0], name, np.std(L), more_stats_d)



def make_profile_plot():

    # {{{}}}
    pass # delete



def make_scatter_plot():

    # {{{}}}
    pass # delete



def plot_pca(y, pca, yp, L, name):
	lx = np.linspace(np.min(L), np.max(L), 100)
	x = np.linspace(0,1,y.shape[1])
	plt.subplot(231)
	plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
	plt.subplot(232)
	m,b,r,p,s = stats.linregress(L, yp[:,0])
	plt.scatter(L, yp[:,0])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(233)
	m,b,r,p,s = stats.linregress(L, yp[:,1])
	plt.scatter(L, yp[:,1])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(234)
	plt.plot(x, np.exp(y[:100].T), alpha=0.5)
	plt.title('data')
	plt.subplot(235)
	try:
		n_samples = 50
		s = np.random.normal(scale=np.std(yp[:,0]), size=n_samples)
		v = np.vstack([s, np.zeros(n_samples)]).T
		yt = pca.inverse_transform(v)
		plt.plot(x, np.exp(yt.T), alpha=0.5)
		plt.title('pc1')
		plt.subplot(236)
		n_samples = 50
		s = np.random.normal(scale=np.std(yp[:,1]), size=n_samples)
		v = np.vstack([np.zeros(n_samples), s]).T
		yt = pca.inverse_transform(v)
		plt.plot(x, np.exp(yt.T), alpha=0.5)
		plt.title('pc2')
	except ValueError:
		pass
	plt.savefig(scalingBicoidFinalReally.ensure_dir(os.path.join(config.plots_path,
        "presentation", "pca_presentation_fig.pdf")))
	plt.clf()



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

    # Run PCA analysis
    pca, res = do_pca_analysis(scaling_profiles, scaling_lengths, 'scaling_large')