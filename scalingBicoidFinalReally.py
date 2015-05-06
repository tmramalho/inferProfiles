
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

import config
reload(config)

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
	profile = UnivariateSpline(x, np.log(y), s=0)
	x = np.linspace(0,0.9,900)
	return np.exp(profile(x))

def smooth_arr_prof(p):
	np.place(p, p==0, np.nan)
	mask = np.isnan(p)
	p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
	return p[:900]

def smooth_arr_bg(p,t):
	mask = np.isnan(p)
	p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
	y1 = 600.0
	y2 = 15.0
	if t == 0:
		x1 = 662
		x2 = 180
	elif t == 1:
		x1 = 724
		x2 = 180
	else:
		x1 = 703
		x2 = 174
	a = (y1 - y2) / (x1 - x2)
	b = (x1 * y2 - x2 * y1) / (x1 - x2)
	s = lambda x: np.clip(a*x+b, 3, None)
	return s(p[:900])

def plot_pca(y, pca, yp, L, name):
	lx = np.linspace(np.min(L), np.max(L), 100)
	x = np.linspace(0,0.9,y.shape[1])
	plt.subplot(231)
	plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
	plt.subplot(232)
	m,b,r,p,s = stats.linregress(L, yp[:,0])
	plt.scatter(L, yp[:,0])
	plt.plot(lx, m*lx+b, color='k')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(233)
	m,b,r,p,s = stats.linregress(L, yp[:,1])
	plt.scatter(L, yp[:,1])
	plt.plot(lx, m*lx+b, color='k')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(234)
	jet = plt.get_cmap('jet')
	norm = colors.Normalize(vmin=np.min(L), vmax=np.max(L))
	for i in np.random.choice(y.shape[0], np.min([100, y.shape[0]]), replace=False):
		plt.plot(x, np.exp(y[i]), c=jet(norm(L[i])), alpha=0.8)
	plt.title('data')
	plt.subplot(235)
	try:
		n_samples = 50
		s = np.random.normal(scale=np.std(yp[:,0]), size=n_samples)
		v = np.vstack([s, np.zeros(n_samples)]).T
		yt = pca.inverse_transform(v)
		plt.plot(x, np.exp(yt.T), c='b', alpha=0.3)
		plt.title('pc1')
		plt.subplot(236)
		n_samples = 50
		s = np.random.normal(scale=np.std(yp[:,1]), size=n_samples)
		v = np.vstack([np.zeros(n_samples), s]).T
		yt = pca.inverse_transform(v)
		plt.plot(x, np.exp(yt.T), c='b', alpha=0.3)
		plt.title('pc2')
	except ValueError:
		pass
	plt.savefig(ensure_dir(os.path.join(config.plots_path, "bcdpca", "{0}.pdf".format(name))))
	plt.clf()

def do_pca_analysis(profiles, lens, name='', pca=None, plot=True, print_debug=False):
	L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
	pr = []
	for i,p in enumerate(profiles):
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		av, va = moving_average(np.log(p+0.001), 46, 100)
		pr.append(av)
	y = np.array(pr)
	if pca is None:
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
	if plot or name.startswith('LEandSE'):
		plot_pca(y, pca, yp, L, name)
	more_stats_d = {'norm_sigma_l': np.std(lens) / np.mean(lens)}
	return pca, (r1, p1, r2, p2, L.shape[0], name, np.std(L), more_stats_d)

def plot_all_pcs(profiles):
	pr = []
	for i,p in enumerate(profiles):
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		av, va = moving_average(np.log(p+0.001), 46, 100)
		pr.append(av)
	y = np.array(pr)
	pca = PCA(n_components=6)
	pca.fit(y)
	yp = pca.transform(y)
	x = np.linspace(0,0.9,y.shape[1])
	n_samples = 50
	plt.figure(figsize=(8, 9))
	for i in xrange(6):
		plt.subplot(3,2,i+1)
		s = np.random.normal(scale=np.std(yp[:,i]), size=n_samples)
		v = np.zeros((6, n_samples))
		v[i] = s
		yt = pca.inverse_transform(v.T)
		plt.plot(x, np.exp(yt.T), c='b', alpha=0.3)
		plt.ylabel('PC{0:d}'.format(i+1))
		plt.xlabel('AP position (x/L)')
	plt.tight_layout()
	plt.savefig('plots/SI_pcgrid.pdf')
	plt.clf()

def plot_ks_analysis(lower_y, upper_y, pval, name):
	plt.figure(figsize=(5,10))
	plt.subplot(311)
	ax = plt.gca()
	ax.set_axis_bgcolor('white')
	plt.title('smaller embryos, N={0}'.format(lower_y.shape[0]))
	for i,p in enumerate(pval):
		if p < 0.01:
			plt.axvspan(i, i+1, facecolor='0.5', alpha=0.5)
		elif p < 0.05:
			plt.axvspan(i, i+1, facecolor='0.5', alpha=0.2)
	plt.plot(np.exp(lower_y.T[:,:100]))
	plt.subplot(312)
	plt.title('bigger embryos, N={0}'.format(upper_y.shape[0]))
	plt.plot(np.exp(upper_y.T[:,:100]))
	plt.subplot(313)
	plt.plot(pval)
	plt.plot(np.ones(shape=len(pval))*0.05, "g--", label=r"5\% confidence")
	plt.plot(np.ones(shape=len(pval))*0.01, "r--", label=r"1\% confidence")
	plt.yscale('log')
	plt.ylabel('pvalue', fontsize=10)
	plt.xlabel('x/L')
	plt.suptitle(name)
	plt.savefig(ensure_dir(os.path.join(config.plots_path, "plots", "bcdpca", "ks_{0}.pdf".format(name))))
	plt.clf()

def do_ks_analysis(profiles, lens, name='', plot=False):
	L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
	n, bins = np.histogram(L, bins=2)
	idx_l = np.digitize(L, bins)
	pos_l = (idx_l == 1)
	r_list_l = np.where(pos_l)[0]
	print profiles.shape, profiles.dtype
	lower_y = np.log(np.array(list(profiles[r_list_l]), dtype=np.float))
	# print lower_y.shape, L[r_list_l]
	pos_u = (idx_l > 1)
	r_list_u = np.where(pos_u)[0]
	upper_y = np.log(np.array(list(profiles[r_list_u]), dtype=np.float))
	# print upper_y.shape, L[r_list_u]
	if upper_y.shape[0] < 2 or lower_y.shape[0] < 2:
		return np.ones(profiles[0].shape[0])
	'''ks 2 sample'''
	pval=[]
	for k in xrange(lower_y.shape[1]):
		try:
			_, p = stats.ks_2samp(lower_y[:,k], upper_y[:,k])
		except ValueError:
			p = 1
		pval.append(p)
	if plot:
		plot_ks_analysis(lower_y, upper_y, pval, name)
	pv = np.array(pval)
	return pv

def make_summary_plot(res, ks_res, min_sample_size=11):
	plt.figure(figsize=(4,3.5))
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
	plt.title('summary of bicoid data')
	plt.ylabel('r_sq')
	plt.xlabel('sigma_l')
	plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 'bcd_bubbles.pdf')))
	plt.clf()
	plt.figure(figsize=(8,9))
	for i,dataset in enumerate(res):
		if dataset[4] > min_sample_size:
			p1 = np.mean(dataset[1])
			p2 = np.mean(dataset[3])
			r1 = np.mean(np.power(dataset[0],2))
			r2 = np.mean(np.power(dataset[2],2))
			s_size = dataset[4]
			sigma_l = dataset[6]
			plt.subplot(3,2,1)
			plt.scatter(s_size, p1)
			plt.ylabel('p-value')
			plt.xlabel('sample size')
			plt.subplot(3,2,2)
			plt.scatter(s_size, p2)
			plt.ylabel('p-value')
			plt.xlabel('sample size')
			plt.subplot(3,2,3)
			plt.scatter(sigma_l, p1)
			plt.ylabel('p-value')
			plt.xlabel('length variation')
			plt.subplot(3,2,4)
			plt.scatter(sigma_l, p2)
			plt.ylabel('p-value')
			plt.xlabel('length variation')
			plt.subplot(3,2,5)
			plt.scatter(r1, p1)
			plt.ylabel('p-value')
			plt.xlabel('r^2')
			plt.subplot(3,2,6)
			plt.scatter(r2, p2)
			plt.ylabel('p-value')
			plt.xlabel('r^2')
	plt.tight_layout()
	plt.savefig('plots/summary/si_params.pdf')
	plt.clf()

def make_table(res, ks_res, min_sample_size=11):
	labels = []
	ks_ratio = []
	pca_vals = []
	pca_std = []
	s_size = []
	sigma_l = []
	r_sq = []
	r_ci = []
	for i,dataset in enumerate(res):
		if dataset[4] > min_sample_size and dataset[6]> 5:
			labels.append(dataset[5])
			pv = ks_res[i]
			reject_ratio1 = np.where(pv<0.01)[0].shape[0] / float(pv.shape[0])
			p1 = np.mean(dataset[1])
			p2 = np.mean(dataset[3])
			pca_vals.append([p1, p2])
			p1v = [np.percentile(dataset[1], 2.5), np.percentile(dataset[1], 97.5)]
			p2v = [np.percentile(dataset[3], 2.5), np.percentile(dataset[3], 97.5)]
			pca_std.append([p1v, p2v])
			r1 = np.mean(np.power(dataset[0],2))
			r2 = np.mean(np.power(dataset[2],2))
			r_sq.append([r1, r2])
			r1v = [np.percentile(np.power(dataset[0],2), 2.5), np.percentile(np.power(dataset[0],2), 97.5)]
			r2v = [np.percentile(np.power(dataset[2],2), 2.5), np.percentile(np.power(dataset[2],2), 97.5)]
			r_ci.append([r1v,r2v])
			s_size.append(dataset[4])
			sigma_l.append(dataset[6])
			reject_ratio2 = np.where(pv<0.05)[0].shape[0] / float(pv.shape[0])
			ks_ratio.append((reject_ratio1, reject_ratio2))
	pca_vals = np.array(pca_vals)
	pca_std = np.array(pca_std)
	r_sq = np.array(r_sq)
	r_ci = np.array(r_ci)

	def saturate_pvals(inp):
		out = np.empty_like(inp)
		out[np.where(inp <= 0.05)] = 1
		out[np.where(inp > 0.05)] = -1
		return 0.5*out

	num_cols = len(labels)
	color_data = np.vstack([
		np.zeros((num_cols,)),
		saturate_pvals(np.array(pca_vals[:,0])),
	    saturate_pvals(np.array(pca_vals[:,1])),
		np.zeros((num_cols,)),
	    np.zeros((num_cols,)),
	    np.zeros((num_cols,)),
	    np.zeros((num_cols,))])
	num_rows = color_data.shape[0]
	plt.figure(figsize=(20, 7))
	ax = plt.gca()
	ax.imshow(color_data, interpolation='nearest', cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

	#text portion
	x_array = np.arange(num_cols)
	y_array = np.arange(num_rows)
	x, y = np.meshgrid(x_array, y_array)
	print num_rows, num_cols

	for x_val, y_val in zip(x.flatten(), y.flatten()):
		if y_val == 0:
			tx = '{0:.02f} (1%)\n{1:.02f} (5%)'.format(ks_ratio[x_val][0], ks_ratio[x_val][1])
			ax.text(x_val, y_val, tx, va='center', ha='center')
		elif y_val == 1:
			tx = '{0:.01e}\n[{1:.02e},\n{2:.02e}]'.format(pca_vals[x_val,0], pca_std[x_val,0,0], pca_std[x_val,0,1])
			ax.text(x_val, y_val, tx, va='center', ha='center')
		elif y_val == 2:
			tx = '{0:.01e}\n[{1:.02e},\n{2:.02e}]'.format(pca_vals[x_val,1], pca_std[x_val,1,0], pca_std[x_val,1,1])
			ax.text(x_val, y_val, tx, va='center', ha='center')
		elif y_val == 3:
			tx = '{0}'.format(s_size[x_val])
			ax.text(x_val, y_val, tx, va='center', ha='center')
		elif y_val == 4:
			tx = '{0:.02f}'.format(sigma_l[x_val])
			ax.text(x_val, y_val, tx, va='center', ha='center')
		elif y_val == 5:
			tx = '{0:.02f}\n[{1:.02f},{2:.02f}]'.format(r_sq[x_val,0], r_ci[x_val,0,0], r_ci[x_val,0,1])
			ax.text(x_val, y_val, tx, va='center', ha='center')
		elif y_val == 6:
			tx = '{0:.02f}\n[{1:.02f},{2:.02f}]'.format(r_sq[x_val,1], r_ci[x_val,1,0], r_ci[x_val,1,1])
			ax.text(x_val, y_val, tx, va='center', ha='center')

	ax.set_xticks(x_array+0.5)
	ax.set_yticks(y_array+0.5)
	ax.set_xticklabels(labels, rotation = 90)
	ax.set_yticklabels(['ks', 'pca1', 'pca2', 'n_samples', 'sigma_l', 'r_sq1', 'r_sq2'])
	plt.tight_layout()
	plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 'bcd_table.pdf')))
	plt.clf()

	# ks_ratio = np.array(ks_ratio)
	# plt.figure(figsize=(4,10))
	# plt.subplot(411)
	# plt.bar(np.arange(len(pca_vals)), ks_ratio[:,0] , 0.5)
	# plt.bar(np.arange(len(pca_vals))+0.5, ks_ratio[:,1] , 0.5, color='r')
	# plt.subplot(412)
	# plt.bar(np.arange(len(pca_vals)), pca_vals, 1)
	# plt.subplot(413)
	# plt.bar(np.arange(len(s_size)), s_size, 1)
	# plt.subplot(414)
	# plt.bar(np.arange(len(sigma_l)), sigma_l, 1)
	# plt.tight_layout()
	# plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 'bcd_bar.pdf')))
	# plt.clf()


if __name__ == '__main__':
	try:
		(results, ks_results) = np.load(os.path.join(config.tmp_path, 'results.npy'))
	except IOError:
		results = []
		ks_results = []

		'''
		Load 2XA data
		'''
		bcd_data = sio.loadmat(os.path.join(config.scaling_data_path, 
                                               '2XAAllEmbryos.mat'), squeeze_me=True)
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
		Load scaling data
		'''
		bcd_sdata = sio.loadmat(os.path.join(config.scaling_data_path,
                                                'ScalingDataLargeSetBcd.mat'),
                                   squeeze_me=True)
		bcd_sdata = bcd_sdata['RawData']['M'].item()['Em'].item()
		scaling_lengths = []
		scaling_profiles = []
		for p,l in zip(bcd_sdata[:]['Profile'], bcd_sdata[:]['EL']):
			scaling_lengths.append(l)
			scaling_profiles.append(smooth_arr_prof(p))
		scaling_profiles = np.array(scaling_profiles)
		scaling_lengths = np.array(scaling_lengths)

		'''
		GFP session largest
		'''
		gfp_data = sio.loadmat(os.path.join(config.scaling_data_path,
                                               'ScalingDataLargestBcdGFPSession.mat'),
                                  squeeze_me=True)
		dat = gfp_data['RawData']['M'].item()['Em'].item()
		GFP_L = dat['EL'].astype('float64')
		GFP_profiles = []
		for p in dat['Profile']:
			GFP_profiles.append(smooth_arr_prof(p))
		GFP_profiles = np.array(GFP_profiles)

		'''
		Same analysis for temp varied dataset
		'''
		tmp_data = sio.loadmat(os.path.join(config.scaling_data_path,
                                               'ScalingDataTempVariedBcd.mat'),
                                  squeeze_me=True)
		dat = tmp_data['RawData'].item()[0]['Em'].item()
		temp_L = dat['EL'].astype('float64')
		temp_profiles = []
		for p in dat['Profile']:
			temp_profiles.append(smooth_arr_prof(p))
		temp_profiles = np.array(temp_profiles)

		'''
		Same analysis for LE and SE embryos
		'''
		dd = sio.loadmat(os.path.join(config.scaling_data_path,
                                         'ScalingData1And23.mat'), squeeze_me=True)
		dat = dd['RawData']['M'].item()
		profiles_symmetric = []
		profiles_ventral = []
		profiles_dorsal = []
		len_symmetric = []
		len_ventral = []
		len_dorsal = []
		for p,or_v,le_v, ag_v in zip(dat['Em'][1]['Profile'], dat['Em'][1]['orientation'], dat['Em'][1]['EL'], dat['Em'][1]['Emage']):
			# age binning
			#if ag_v < 140 or ag_v > 175:
			#	continue
			# outlier removal
			if np.any(p[:100,1,0] < 200):
				continue
			if or_v == 0:
				profiles_symmetric.append(smooth_arr_bg(p[:,1,0], t=0))
				len_symmetric.append(le_v)
				profiles_symmetric.append(smooth_arr_bg(p[:,1,1], t=0))
				len_symmetric.append(le_v)
			else:
				profiles_ventral.append(smooth_arr_bg(p[:,1,1], t=1))
				len_ventral.append(le_v)
				profiles_dorsal.append(smooth_arr_bg(p[:,1,0], t=2))
				len_dorsal.append(le_v)
		profiles_symmetric = np.array(profiles_symmetric)
		profiles_ventral = np.array(profiles_ventral)
		profiles_dorsal = np.array(profiles_dorsal)

		'''
		Perform pca analysis on all profiles, then randomly subsample 150 and 300 from
		the whole set
		'''
		combo_profiles = np.concatenate([scaling_profiles, GFP_profiles, temp_profiles, profiles_dorsal, profiles_symmetric, profiles_ventral])
		combo_lengths = np.concatenate([scaling_lengths, GFP_L, temp_L, len_dorsal, len_symmetric, len_ventral])
		plot_all_pcs(combo_profiles)
		pca, res = do_pca_analysis(combo_profiles, combo_lengths, 'MEGACOMBO')
		pvals = do_ks_analysis(combo_profiles, combo_lengths, 'MEGACOMBO')
		# results.append(res)
		pca, res = do_pca_analysis(scaling_profiles, scaling_lengths, 'scaling_large', pca)
		results.append(res)
		pvals = do_ks_analysis(scaling_profiles, scaling_lengths, 'scaling_large')
		ks_results.append(pvals)

		sample = np.random.choice(scaling_profiles.shape[0], 100, replace=False)
		pca, res = do_pca_analysis(scaling_profiles[sample], scaling_lengths[sample], 'scaling_large(100)', pca)
		results.append(res)
		pvals = do_ks_analysis(scaling_profiles[sample], scaling_lengths[sample], 'scaling_large(100)')
		ks_results.append(pvals)

		'''
		Perform the same analysis as before, but binned by each individual session
		'''
		for s in np.unique(bcd_sdata[:]['Sessionnum']):
			ind = np.where(bcd_sdata[:]['Sessionnum'] == s)
			pca, res = do_pca_analysis(scaling_profiles[ind], scaling_lengths[ind], 'ind_{0}'.format(s), pca)
			results.append(res)
			pvals = do_ks_analysis(scaling_profiles[ind], scaling_lengths[ind], 'ind_{0}'.format(s))
			ks_results.append(pvals)

		# pca, res = do_pca_analysis(GFP_profiles, GFP_L, 'GFP_session', pca)
		# results.append(res)
		# pvals = do_ks_analysis(GFP_profiles, GFP_L, 'GFP_session')
		# ks_results.append(pvals)
		pca, res = do_pca_analysis(temp_profiles, temp_L, 'Temp_varied', pca)
		results.append(res)
		pvals = do_ks_analysis(temp_profiles, temp_L, 'Temp_varied')
		ks_results.append(pvals)


		pca, res = do_pca_analysis(profiles_symmetric, len_symmetric, 'LEandSE_sym', pca)
		results.append(res)
		pvals = do_ks_analysis(profiles_symmetric, len_symmetric, 'LEandSE_sym')
		ks_results.append(pvals)
		pca, res = do_pca_analysis(profiles_ventral, len_ventral, 'LEandSE_ven', pca)
		results.append(res)
		pvals = do_ks_analysis(profiles_ventral, len_ventral, 'LEandSE_ven')
		ks_results.append(pvals)
		pca, res = do_pca_analysis(profiles_dorsal, len_dorsal, 'LEandSE_dor', pca)
		results.append(res)
		pvals = do_ks_analysis(profiles_dorsal, len_dorsal, 'LEandSE_dor')
		ks_results.append(pvals)

		np.save(ensure_dir(os.path.join(config.tmp_path, 'results.npy')), (results, ks_results))

	make_summary_plot(results, ks_results)
	make_table(results, ks_results)
