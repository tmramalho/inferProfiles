__author__ = 'tiago'
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn.decomposition import PCA
import scipy.stats as stats

def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)
	return f

def moving_average(series, sigma=3, ks=27):
	b = gaussian(ks, sigma)
	average = filters.convolve1d(series, b / b.sum())
	var = filters.convolve1d(np.power(series - average, 2), b / b.sum())
	return average, var

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
	plt.savefig(ensure_dir("plots/wt_gap/pca_{0}.pdf".format(name)))
	plt.clf()

def do_pca_analysis(profiles, lens, name='', plot=False):
	L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
	pr = []
	for i,p in enumerate(profiles):
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		av, va = moving_average(np.log(p+0.001), 46, 100)
		pr.append(av)
	y = np.array(pr)
	pca = PCA(n_components=2)
	pca.fit(y)
	print pca.explained_variance_ratio_
	yp = pca.transform(y)
	m,b,r,p,_ = stats.linregress(L, yp[:,0])
	p1 = [p]
	r1 = [r]
	for _ in xrange(300):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[~sample], yp[~sample,0])
		p1.append(p)
		r1.append(r)
	m,b,r,p,_ = stats.linregress(L, yp[:,1])
	p2 = [p]
	r2 = [r]
	for _ in xrange(300):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[~sample], yp[~sample,1])
		p2.append(p)
		r2.append(r)
	if plot:
		plot_pca(y, pca, yp, L, name)
	return r1, p1, r2, p2, L.shape[0], name, np.std(L)

def plot_ks_analysis(lower_y, upper_y, pval, name):
	plt.grid(False, which='both')
	plt.subplot(311)
	plt.plot(lower_y.T)
	plt.subplot(312)
	plt.plot(upper_y.T)
	plt.subplot(313)
	plt.plot(pval)
	plt.plot(np.ones(shape=len(pval))*0.05, "g--", label=r"5\% confidence")
	plt.plot(np.ones(shape=len(pval))*0.01, "r--", label=r"1\% confidence")
	plt.yscale('log')
	plt.ylabel('pvalue', fontsize=10)
	plt.xlabel('x/L')
	plt.suptitle(name)
	plt.savefig(ensure_dir("plots/wt_gap/ks_{0}.pdf".format(name)))
	plt.clf()
	plt.grid(True, which='both')

def do_ks_analysis(profiles, lens, name='', plot=False):
	L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
	n, bins = np.histogram(L, bins=2)
	idx_l = np.digitize(L, bins)
	pos_l = (idx_l == 1)
	r_list_l = np.where(pos_l)[0]
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

def nan_helper(y):
	"""Helper to handle indices and logical indices of NaNs.

	Input:
		- y, 1d numpy array with possible NaNs
	Output:
		- nans, logical indices of NaNs
		- index, a function, with signature indices= index(logical_indices),
		  to convert logical indices of NaNs to 'equivalent' indices
	Example:
		# linear interpolation of NaNs
		nans, x= nan_helper(y)
		y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	"""

	return np.isnan(y), lambda z: z.nonzero()[0]

class ScalingContainer(object):

	def __init__(self):
		print 'Reading matlab...'
		dd = sio.loadmat('data/scaling_data/DataSets/ScalingData1And23.mat', squeeze_me=True)
		dat = dd['RawData']['M'].item()
		print 'Generating data...'
		self.data_containers = []
		for i in range(2):
			d1 = dict()
			d1['Flylinename'] = dat['Flylinename'][i] + str(i)
			d1['Genename'] = dat['Genename'][i]
			c = np.concatenate([aux[..., np.newaxis] for aux in dat['Em'][i]['Profile']], axis=3)
			d1['meas'] = c.astype('float64')
			d1['age'] = dat['Em'][i]['Emage'].astype('float64')
			d1['L'] = dat['Em'][i]['EL'].astype('float64')
			d1['nc'] = dat['Em'][i]['nc'].astype('float64')
			d1['or'] = dat['Em'][i]['orientation'].astype('int')
			#get rid of missing ages
			d1['meas'] = d1['meas'][:,:,:,np.isfinite(d1['age'])]
			d1['L'] = d1['L'][np.isfinite(d1['age'])]
			d1['nc'] = d1['nc'][np.isfinite(d1['age'])]
			d1['age'] = d1['age'][np.isfinite(d1['age'])]
			d1['or'] = d1['or'][np.isfinite(d1['age'])]
			#get rid of outlier
			# if i == 0:
			# 	outliers = []
			# 	for n,_ in enumerate(d1['L']):
			# 		if np.any(np.log(d1['meas'][:,:,:,n]) < 4.5):
			# 			outliers.append(n)
			# else:
			# 	outliers = []
			# 	for n,_ in enumerate(d1['L']):
			# 		if np.any(np.log(d1['meas'][:,1:,:,n]) < 4.5):
			# 			outliers.append(n)
			# print outliers
			# d1['meas'] =  np.delete(d1['meas'], outliers, axis=3)
			# d1['L'] = np.delete(d1['L'], outliers)
			# d1['nc'] = np.delete(d1['nc'], outliers)
			# d1['age'] = np.delete(d1['age'], outliers)
			# d1['or'] = np.delete(d1['or'], outliers)
			self.data_containers.append(d1)
		print 'Done.'

	def bin_orientation(self, nc, or_v, e):
		if nc > 13:
			if or_v == 2:
				oi = 0
			elif or_v and e == 1:
				oi = 1
			else:
				oi = 2
		else:
			if or_v == 0:
				oi = 0
			elif or_v == 1 and e == 1:
				oi = 1
			else:
				oi = 2
		return oi

	def time_plot(self, di=20):
		for li in xrange(2):
			for dc in self.data_containers:
				nc = dc['nc'][0]
				Lav = dc['L'].mean()
				for i, g in enumerate(dc['Genename']):
					print g, i, dc['Flylinename']
					n_bins = 20
					n, bins = np.histogram(dc['age'], bins=n_bins)
					idx = np.digitize(dc['age'], bins)
					yav = []
					for bi in xrange(n_bins):
						pos = (idx == (bi+1))
						r_list = np.where(pos)[0]
						y = [[],[],[]]
						for r in r_list:
							for e in xrange(2):
								oi = self.bin_orientation(nc, dc['or'][r], e)
								if di > 0:
									pr = dc['meas'][di:-di,i,e,r]
								else:
									pr = dc['meas'][:,i,e,r]
								nans, xp = nan_helper(pr)
								pr[nans] = np.interp(xp(nans), xp(~nans), pr[~nans])
								if li == 0:
									prd = pr
								else:
									prd = np.log(pr)
								if nc > 13:
									av, _ = moving_average(prd, 8)
								else:
									av, _ = moving_average(prd, 26, ks=47)
								y[oi].append(av)
						m = []
						for k in xrange(3):
							if len(y[k]) != 0:
								m1 = np.mean(np.array(y[k]), axis=0)
							else:
								m1 = np.zeros((1000-2*di))
							m.append(m1)
						yav.append(m)
					jet = plt.get_cmap('jet')
					gs = plt.GridSpec(3,2, width_ratios=[10,1])
					for k in xrange(3):
						plt.subplot(gs[k, 0])
						for j in xrange(n_bins):
							pr = yav[j][k]
							if np.max(pr) < 0.1:
								continue
							x = np.linspace(0,1,pr.shape[0])
							plt.plot(x, pr, c=jet(float(j)/n_bins), label=bins[j])
						if k == 0:
							plt.title('Symmetric')
						elif k == 1:
							plt.title('Ventral')
						elif k == 2:
							plt.title('Dorsal')
					sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=0, vmax=n_bins))
					sm._A = []
					plt.suptitle("{0} {1}".format(dc['Flylinename'], g))
					plt.colorbar(sm, cax=plt.subplot(gs[:, 1]))
					plt.savefig(ensure_dir("plots/wt_gap/timeplot_{0}_{1}_{2}.pdf".format(dc['Flylinename'], g, li)))
					plt.clf()

	def analysis(self, di=20, plot=False):
		names = ['Symmetric', 'Ventral', 'Dorsal']
		for dc in self.data_containers:
			nc = dc['nc'][0]
			Lav = dc['L'].mean()
			for i, g in enumerate(dc['Genename']):
				for bi in xrange(2):
					if bi == 0:
						bi_name = 'early'
						pos = (dc['age'] >= 145) & (dc['age'] <= 155)
					else:
						bi_name = 'late'
						pos = (dc['age'] >= 170) & (dc['age'] <= 180)
					r_list = np.where(pos)[0]
					y_arr = [[], [], []]
					ll = [[], [], []]
					a = [[], [], []]
					for r in r_list:
						for e in xrange(2):
							oi = self.bin_orientation(nc, dc['or'][r], e)
							if di > 0:
								pr = dc['meas'][di:-di,i,e,r]
							else:
								pr = dc['meas'][:,i,e,r]
							nans, xp = nan_helper(pr)
							pr[nans] = np.interp(xp(nans), xp(~nans), pr[~nans])
							y_arr[oi].append(pr)
							ll[oi].append(dc['L'][r])
							a[oi].append(dc['age'][r])
					for oi in xrange(3):
						y = np.array(y_arr[oi])
						L = np.array(ll[oi]) - Lav
						if len(y.shape) < 2:
							print "Few samples!"
							continue
						name = "{0}_{1}_{2}_{3}".format(dc['Flylinename'], g, names[oi], bi_name)
						_,p1,_,p2,_,_,_ = do_pca_analysis(y, L, name, plot)
						pv = do_ks_analysis(y, L, name, plot)
						'''
						Rejection analysis
						'''
						reject_ratio = np.where(pv<0.01)[0].shape[0] / float(pv.shape[0])
						p1m = np.mean(p1)
						p2m = np.mean(p2)
						print name, y.shape[0]
						if (p1m < 0.01 or p2m < 0.01) and reject_ratio > 0.1:
							print 'REJECT'
							print p1m, p2m, reject_ratio


if __name__ == '__main__':
	sc = ScalingContainer()
	sc.analysis(plot=True)