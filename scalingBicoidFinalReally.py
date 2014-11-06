
from __future__ import division
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.interpolate import UnivariateSpline
import numpy as np
import os
from matplotlib import rc
import matplotlib.pyplot as plt
import LiuHugeDatasetProcess as ldp
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn.decomposition import PCA
import scipy.stats as stats

def moving_average(series, sigma=3, ks=27):
	b = gaussian(ks, sigma)
	average = filters.convolve1d(series, b / b.sum())
	var = filters.convolve1d(np.power(series - average, 2), b / b.sum())
	return average, var

def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)
	return f

def smooth_nuc_prof(p):
	x = np.concatenate([[0], p[0], [1]])
	y = np.concatenate([[p[1,0]], p[1], [p[1,-1]]])
	profile = UnivariateSpline(x, np.log(y), s=100)
	x = np.linspace(0,0.9,1000)
	return profile(x)

def smooth_arr_prof(p):
	mask = np.isnan(p)
	p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
	profile = UnivariateSpline(np.linspace(0,1,p.shape[0]), np.log(p), s=100)
	x = np.linspace(0,0.9,1000)
	return profile(x)

def smooth_arr_bg(p):
	m = (3000-5)/(3000-150)
	s = lambda x: np.abs(5 + m*(x-150))
	mask = np.isnan(p)
	p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
	profile = UnivariateSpline(np.linspace(0,1,p.shape[0]), np.log(s(p)), s=1000)
	x = np.linspace(0,0.9,1000)
	return profile(x)

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
	plt.plot(x, y.T, alpha=0.5)
	plt.title('data')
	plt.subplot(235)
	n_samples = 50
	s = np.random.normal(scale=np.std(yp[:,0]), size=n_samples)
	v = np.vstack([s, np.zeros(n_samples)]).T
	yt = pca.inverse_transform(v)
	plt.plot(x, yt.T, alpha=0.5)
	plt.title('pc1')
	plt.subplot(236)
	n_samples = 50
	s = np.random.normal(scale=np.std(yp[:,1]), size=n_samples)
	v = np.vstack([np.zeros(n_samples), s]).T
	yt = pca.inverse_transform(v)
	plt.plot(x, yt.T, alpha=0.5)
	plt.title('pc2')
	plt.savefig(ensure_dir("plots/bcdpca/{0}.pdf".format(name)))
	plt.clf()

def do_pca_analysis(profiles, lens, name='', pca=None, plot=False):
	print 'PCAing', name
	L = lens-np.mean(lens)
	print L.shape
	y = np.array(profiles)
	if pca is None:
		pca = PCA(n_components=2)
		pca.fit(y)
		print pca.explained_variance_ratio_
	yp = pca.transform(y)
	m,b,r1,p1,_ = stats.linregress(L, yp[:,0])
	m,b,r2,p2,_ = stats.linregress(L, yp[:,1])
	if plot:
		plot_pca(y, pca, yp, L, name)
	return pca, (name, L.shape[0], p1, r1, p2, r2, np.std(L))

def make_bubble_plot(df, x, y, z, min_sample_size=12):
	ax = plt.gca()
	labels = []
	cols = []
	ind_start = False
	for i in df.index:
		if not df.loc[i,'Linename'].startswith('ind'):
			next_col = ax._get_lines.color_cycle.next()
			labels.append(df.loc[i,'Linename'])
			cols.append(next_col)
		elif not ind_start:
			next_col = ax._get_lines.color_cycle.next()
			labels.append('2XA ind. sessions')
			cols.append(next_col)
			ind_start = True
		if df.loc[i, 'n'] > min_sample_size:
			plt.scatter(df.loc[i, x], np.abs(df.loc[i, y]), s=df.loc[i, z].astype('int'),
	            alpha=0.5, lw=0, c=next_col)
	plt.plot((0.01,0.01), (df[y].abs().min(), df[y].abs().max()), 'r--', alpha=0.6)
	plt.plot((0.05,0.05), (df[y].abs().min(), df[y].abs().max()), 'g--', alpha=0.6)
	plt.xscale('log')
	plt.xlim([df[x].min(), df[x].max()])
	plt.ylim([df[y].abs().min(), df[y].abs().max()])
	plt.xlabel(x)
	plt.ylabel(y)
	rects = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in cols]
	leg = plt.legend(rects, labels, loc='upper left')
	leg.get_frame().set_alpha(0.5)
	plt.savefig(ensure_dir('plots/summary/bubble_{0}_{1}_{2}.pdf'.format(x,y,z)))
	plt.clf()

if __name__ == '__main__':
	results = []

	'''
	Load 2XA data
	'''
	bcd_data = sio.loadmat('data/scaling_data/DataSets/2XAAllEmbryos.mat', squeeze_me=True)
	bcd_data = bcd_data['AllEmbryos_2XA']
	sessions = np.unique(bcd_data['Session'])
	quality = 1

	'''
	Select all embryos with certain quality
	'''
	targ_embryos = np.where((bcd_data['Quality'] <= quality))
	lengths = bcd_data['EggLength'][targ_embryos].astype('float64')
	raw_profiles = bcd_data['Gradient'][targ_embryos]
	left_profiles = []
	left_lengths = []
	right_profiles = []
	right_lengths = []
	for p,l in zip(raw_profiles, lengths):
		pr = p['left'].item()
		if np.all(pr[:, 1] > 1):
			left_profiles.append(smooth_nuc_prof(np.transpose(pr)))
			left_lengths.append(l)
		pr = p['right'].item()
		if np.all(pr[:, 1] > 1):
			right_profiles.append(smooth_nuc_prof(np.transpose(pr)))
			right_lengths.append(l)
	all_profiles = np.array(right_profiles + left_profiles)
	all_lengths = np.array(right_lengths + left_lengths)

	'''
	Perform pca analysis on all profiles, then randomly subsample 150 and 300 from
	the whole set
	'''
	pca, res = do_pca_analysis(all_profiles, all_lengths, '2XAAll_1', plot=True)
	results.append(res)
	sample = np.random.choice(all_profiles.shape[0], 150, replace=False)
	pca, res = do_pca_analysis(all_profiles[sample], all_lengths[sample], '2XA(150)_1', pca)
	results.append(res)
	sample = np.random.choice(all_profiles.shape[0], 300, replace=False)
	pca, res = do_pca_analysis(all_profiles[sample], all_lengths[sample], '2XA(300)_1', pca)
	results.append(res)

	'''
	Perform the same analysis as before, but binned by each individual session
	'''
	for s in sessions:
		targ_embryos = np.where((bcd_data['Session'] == s) & (bcd_data['Quality'] <= quality))
		raw_profiles = bcd_data['Gradient'][targ_embryos]
		if raw_profiles.shape[0] == 0:
			continue
		lengths = bcd_data['EggLength'][targ_embryos].astype('float64')
		left_profiles = []
		left_lengths = []
		right_profiles = []
		right_lengths = []
		for p,l in zip(raw_profiles, lengths):
			pr = p['left'].item()
			if np.all(pr[:, 1] > 1):
				left_profiles.append(smooth_nuc_prof(np.transpose(pr)))
				left_lengths.append(l)
			pr = p['right'].item()
			if np.all(pr[:, 1] > 1):
				right_profiles.append(smooth_nuc_prof(np.transpose(pr)))
				right_lengths.append(l)
		pca, res = do_pca_analysis(left_profiles, left_lengths, 'ind_{0}_{1}_left'.format(s,quality), pca)
		results.append(res)
		pca, res = do_pca_analysis(right_profiles, right_lengths, 'ind_{0}_{1}_right'.format(s,quality), pca)
		results.append(res)
		pca, res = do_pca_analysis(right_profiles + left_profiles,
		                left_lengths + right_lengths, 'ind_{0}_{1}_both'.format(s,quality-1), pca)
		results.append(res)

	'''
	Same analysis for temp varied dataset
	'''
	bcd_data = sio.loadmat('data/scaling_data/DataSets/ScalingDataTempVariedBcd.mat', squeeze_me=True)
	dat = bcd_data['RawData'].item()[0]['Em'].item()
	L = dat['EL'].astype('float64')
	profiles = []
	for p in dat['Profile']:
		profiles.append(smooth_arr_prof(p))
	pca, res = do_pca_analysis(profiles, L, 'Temp_varied', pca, True)
	results.append(res)

	'''
	Same analysis for LE and SE embryos
	'''
	dd = sio.loadmat('data/scaling_data/DataSets/ScalingData1And23.mat', squeeze_me=True)
	dat = dd['RawData']['M'].item()
	profiles_symmetric = []
	profiles_ventral = []
	profiles_dorsal = []
	len_symmetric = []
	len_ventral = []
	len_dorsal = []
	for p,or_v,le_v, ag_v in zip(dat['Em'][1]['Profile'], dat['Em'][1]['orientation'], dat['Em'][1]['EL'], dat['Em'][1]['Emage']):
		# age binning
		if ag_v < 140 or ag_v > 175:
			continue
		# outlier removal
		if np.any(p[:100,1,0] < 200):
			continue
		if or_v == 0:
			profiles_symmetric.append(smooth_arr_bg(p[:,1,0]))
			len_symmetric.append(le_v)
			profiles_symmetric.append(smooth_arr_bg(p[:,1,1]))
			len_symmetric.append(le_v)
		else:
			profiles_ventral.append(smooth_arr_bg(p[:,1,1]))
			len_ventral.append(le_v)
			profiles_dorsal.append(smooth_arr_bg(p[:,1,0]))
			len_dorsal.append(le_v)
	pca, res = do_pca_analysis(profiles_symmetric, len_symmetric, 'LEandSE_sym', pca, True)
	results.append(res)
	pca, res = do_pca_analysis(profiles_ventral, len_ventral, 'LEandSE_ven', pca, True)
	results.append(res)
	pca, res = do_pca_analysis(profiles_dorsal, len_dorsal, 'LEandSE_dor', pca, True)
	results.append(res)

	'''
	GFP session largest
	'''
	bcd_data = sio.loadmat('data/scaling_data/DataSets/ScalingDataLargestBcdGFPSession.mat', squeeze_me=True)
	dat = bcd_data['RawData']['M'].item()['Em'].item()
	L = dat['EL'].astype('float64')
	profiles = []
	for p in dat['Profile']:
		profiles.append(smooth_arr_prof(p))
	pca, res = do_pca_analysis(profiles, L, 'GFP_session', pca)
	results.append(res)

	np.save(ensure_dir('data/tmp/results.npy'), results)

	df = pd.DataFrame(results, columns=['Linename', 'n', 'p1', 'r1', 'p2', 'r2', 'vL'])
	make_bubble_plot(df, 'p2', 'r2', 'n')
	make_bubble_plot(df, 'p1', 'r1', 'n')
	make_bubble_plot(df, 'p2', 'vL', 'n')
	make_bubble_plot(df, 'p1', 'vL', 'n')
	make_bubble_plot(df, 'p2', 'n', 'vL')
	make_bubble_plot(df, 'p1', 'n', 'vL')
