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
			d1['Flylinename'] = dat['Flylinename'][i]
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
			# 	outliers = [190]
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

	def plot(self):
		for dc in self.data_containers:
			for i, g in enumerate(dc['Genename']):
				print g
				n, bins = np.histogram(dc['age'], bins=20)
				idx = np.digitize(dc['age'], bins)
				for b in xrange(len(bins)-1):
					pos = (idx == (b+1))
					y = dc['meas'][:,i,0,pos]
					if y.shape[1] == 0:
						continue
					y = np.ma.masked_array(y,np.isnan(y))
					x = np.linspace(0,1,y.shape[0])
					plt.subplot(221)
					plt.plot(x,np.log(y), alpha=0.5)
					plt.subplot(222)
					ya = np.mean(np.log(y), axis=1)
					plt.plot(x,ya)
					ys = np.std(np.log(y), axis=1)
					plt.fill_between(x, ya-ys, ya+ys, color='r', alpha=0.2)
					plt.suptitle('{0} {1}'.format(g, np.average(dc['age'][pos])))
					#smoothing splines
					av_p = []
					va_p = []
					r_list = np.where(pos)[0]
					for n in xrange(2):
						for r in r_list:
							pr = dc['meas'][:,i,n,r]
							nans, xp = nan_helper(pr)
							pr[nans] = np.interp(xp(nans), xp(~nans), pr[~nans])
							av, va = moving_average(np.log(pr), 8)
							plt.subplot(223)
							plt.plot(x, av, alpha=0.5)
							plt.fill_between(x, av-np.sqrt(va), av+np.sqrt(va), alpha=0.3)
							plt.subplot(224)
							plt.plot(x, np.sqrt(va), alpha=0.5)
							av_p.append(av)
							va_p.append(va)
					av_p = np.array(av_p)
					va_p = np.array(va_p)
					av_t = np.average(av_p, axis=0)
					va_t = np.median(va_p.flatten())
					xs = np.sum(np.power(np.repeat(av_t, av_p.shape[0]).reshape(av_p.shape) - av_p, 2), axis=0)/va_t
					print va_t
					df = av_p.shape[0]
					#print df, 1-chi2.cdf(xs, df)
					plt.savefig(ensure_dir("plots/scaling_{0}_{1}_{2}.pdf".format(i, g, b)))
					plt.clf()
					plt.plot(x, dc['meas'][:,i,1,r_list[0]])
					plt.plot(x, dc['meas'][:,i,0,r_list[0]])
					plt.savefig(ensure_dir("plots/debug_{0}_{1}_{2}.pdf".format(i, g, b)))
					plt.clf()

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
					print g
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
					plt.colorbar(sm, cax=plt.subplot(gs[:, 1]))
					plt.savefig(ensure_dir("plots/timeplot_{0}_{1}_{2}.pdf".format(int(nc), g, li)))
					plt.clf()

	def pca(self, di=20):
		n_bins = 8
		vals = np.ones((n_bins, 2, 4, 2, 3))
		names = ['Symmetric', 'Ventral', 'Dorsal']
		for li in xrange(2):
			for dc in self.data_containers:
				nc = dc['nc'][0]
				Lav = dc['L'].mean()
				for i, g in enumerate(dc['Genename']):
					print g
					n, bins = np.histogram(dc['age'], bins=n_bins)
					idx = np.digitize(dc['age'], bins)
					sample_size = np.zeros((n_bins, 3))
					for bi in xrange(n_bins):
						pos = (idx == (bi+1))
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
								if li == 0:
									prd = pr
								else:
									prd = np.log(pr)
								if nc > 13:
									av, _ = moving_average(prd, 8)
								else:
									av, _ = moving_average(prd, 26, ks=47)
								y_arr[oi].append(av)
								ll[oi].append(dc['L'][r])
								a[oi].append(dc['age'][r])
						for oi in xrange(3):
							y = np.array(y_arr[oi])
							L = np.array(ll[oi]) - Lav
							if len(y.shape) < 2:
								print "Few samples!"
								vals[bi,0,:,li,oi] = [1,0,0,0]
								vals[bi,1,:,li,oi] = [1,0,0,0]
								continue
							print y.shape
							sample_size[bi, oi] = y.shape[0]
							lx = np.linspace(np.min(L), np.max(L), 100)
							pca = PCA(n_components=2)
							pca.fit(y)
							print pca.explained_variance_ratio_
							yp = pca.transform(y)
							x = np.linspace(0,1,y.shape[1])
							plt.figure(figsize=plt.rcParams['figure.figsize'])
							plt.subplot(231)
							plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
							plt.subplot(232)
							m,b,r,p,s = stats.linregress(L, yp[:,0])
							vals[bi,0,:,li,oi] = [p,m,r,s]
							plt.scatter(L, yp[:,0])
							plt.plot(lx, m*lx+b, color='r')
							plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
							plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
							plt.subplot(233)
							m,b,r,p,s = stats.linregress(L, yp[:,1])
							vals[bi,1,:,li,oi] = [p,m,r,s]
							plt.scatter(L, yp[:,1])
							plt.plot(lx, m*lx+b, color='r')
							plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
							plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r,p))
							plt.subplot(234)
							plt.plot(x, y.T, alpha=0.4)
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
							plt.suptitle('{3} {0} a{1:.1f} n{2}'.format(g, np.average(dc['age'][pos]), y.shape[0], names[oi]))
							plt.savefig(ensure_dir("plots/pca_{0}_{1}_{2}_{3}_{4}.pdf".format(li, int(nc), g, oi, bi)))
							plt.clf()
					plt.figure(figsize=(5,12), tight_layout=True)
					plt.subplot(411)
					for oi in xrange(3):
						plt.plot(vals[:,0,0,li,oi], label='1st pc {0}'.format(names[oi]))
						plt.plot(vals[:,1,0,li,oi], label='2nd pc {0}'.format(names[oi]))
					plt.plot(np.ones(shape=vals.shape[0])*0.05, "g--", label=r"5\% confidence")
					plt.plot(np.ones(shape=vals.shape[0])*0.01, "r--", label=r"1\% confidence")
					plt.title('pval')
					plt.yscale('log')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(412)
					plt.errorbar(np.arange(vals.shape[0]), vals[:,0,1,li,0], yerr=vals[:,0,3,li,0], label='1st pc')
					plt.errorbar(np.arange(vals.shape[0]), vals[:,1,1,li,0], yerr=vals[:,1,3,li,0], label='2nd pc')
					plt.plot(np.zeros(shape=vals.shape[0]), "r--", label=r"no slope")
					plt.title('slope')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(413)
					plt.plot(vals[:,0,2,li,0], label='1st pc')
					plt.plot(vals[:,1,2,li,0], label='2nd pc')
					plt.title('r')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(414)
					for oi in xrange(3):
						plt.plot(sample_size[:,oi], label='{0}'.format(names[oi]))
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.savefig(ensure_dir("plots/tpca_{0}_{1}_{2}.pdf".format(int(nc), g, li)))
					plt.clf()

	def pca_gregor(self, di=20):
		vals = np.ones((2, 2, 4, 2, 3))
		names = ['Symmetric', 'Ventral', 'Dorsal']
		for li in xrange(2):
			for dc in self.data_containers:
				nc = dc['nc'][0]
				Lav = dc['L'].mean()
				#debg
				if nc == 14:
					continue
				for i, g in enumerate(dc['Genename']):
					print g
					if g != 'Bcd':
						continue
					sample_size = np.zeros((2, 3))
					for bi in xrange(2):
						if nc > 13:
							if bi == 0:
								pos = (dc['age'] >= 145) & (dc['age'] <= 155)
							else:
								pos = (dc['age'] >= 170) & (dc['age'] <= 180)
						else:
							if bi == 0:
								pos = (dc['age'] >= 140) & (dc['age'] <= 175)
							else:
								pos = (dc['age'] >= 120) & (dc['age'] <= 190)
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
								if li == 0:
									prd = pr
								else:
									prd = np.log(pr)
								if nc > 13:
									av, _ = moving_average(prd, 8)
								else:
									av, _ = moving_average(prd, 26, ks=47)
								y_arr[oi].append(av)
								ll[oi].append(dc['L'][r])
								a[oi].append(dc['age'][r])
						for oi in xrange(3):
							y = np.array(y_arr[oi])
							L = np.array(ll[oi]) - Lav
							A = np.array(a[oi])
							if len(y.shape) < 2:
								print "Few samples!"
								vals[bi,0,:,li,oi] = [1,0,0,0]
								vals[bi,1,:,li,oi] = [1,0,0,0]
								continue
							print y.shape
							sample_size[bi, oi] = y.shape[0]
							lx = np.linspace(np.min(L), np.max(L), 100)
							pca = PCA(n_components=2)
							pca.fit(y)
							print pca.explained_variance_ratio_
							yp = pca.transform(y)
							x = np.linspace(0,1,y.shape[1])
							plt.figure(figsize=plt.rcParams['figure.figsize'])
							plt.subplot(231)
							plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
							plt.subplot(232)
							m,b,r,p,s = stats.linregress(L, yp[:,0])
							vals[bi,0,:,li,oi] = [p,m,r,s]
							plt.scatter(L, yp[:,0])
							plt.plot(lx, m*lx+b, color='r')
							plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
							plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
							plt.subplot(233)
							m,b,r,p,s = stats.linregress(L, yp[:,1])
							vals[bi,1,:,li,oi] = [p,m,r,s]
							plt.scatter(L, yp[:,1])
							plt.plot(lx, m*lx+b, color='r')
							plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
							plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r,p))
							plt.subplot(234)
							plt.plot(x, y.T, alpha=0.4)
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
							plt.suptitle('{3} {0} a{1:.1f} n{2}'.format(g, np.average(dc['age'][pos]), y.shape[0], names[oi]))
							plt.savefig(ensure_dir("plots/wt/gr_pca_{0}_{1}_{2}_{3}_{4}_{5}.pdf".format(li, dc['Flylinename'], int(nc), g, oi, bi)))
							plt.clf()
							plt.subplot(211)
							plt.scatter(A, yp[:,0])
							plt.subplot(212)
							plt.scatter(A, yp[:,1])
							plt.savefig(ensure_dir("plots/wt/gr_pca_{0}_{1}_{2}_{3}_{4}_{5}_age.pdf".format(li, dc['Flylinename'], int(nc), g, oi, bi)))
							plt.clf()
					plt.figure(figsize=(5,12), tight_layout=True)
					plt.subplot(411)
					for oi in xrange(3):
						plt.plot(vals[:,0,0,li,oi], label='1st pc {0}'.format(names[oi]))
						plt.plot(vals[:,1,0,li,oi], label='2nd pc {0}'.format(names[oi]))
					plt.plot(np.ones(shape=vals.shape[0])*0.05, "g--", label=r"5\% confidence")
					plt.plot(np.ones(shape=vals.shape[0])*0.01, "r--", label=r"1\% confidence")
					plt.ylim([0.0001, 1.1])
					plt.title('pval')
					plt.yscale('log')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(412)
					plt.errorbar(np.arange(vals.shape[0]), vals[:,0,1,li,0], yerr=vals[:,0,3,li,0], label='1st pc')
					plt.errorbar(np.arange(vals.shape[0]), vals[:,1,1,li,0], yerr=vals[:,1,3,li,0], label='2nd pc')
					plt.plot(np.zeros(shape=vals.shape[0]), "r--", label=r"no slope")
					plt.title('slope')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(413)
					plt.plot(vals[:,0,2,li,0], label='1st pc')
					plt.plot(vals[:,1,2,li,0], label='2nd pc')
					plt.title('r')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(414)
					for oi in xrange(3):
						plt.plot(sample_size[:,oi], label='{0}'.format(names[oi]))
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.savefig(ensure_dir("plots/wt/gr_tpca_{0}_{1}_{2}_{3}.pdf".format(dc['Flylinename'], int(nc), g, li)))
					plt.clf()

	def pca_maternal(self, di=20):
		names = ['Symmetric', 'Ventral', 'Dorsal']
		for li in xrange(2):
			dc = self.data_containers[1]
			nc = dc['nc'][0]
			Lav = dc['L'].mean()
			for i, g in enumerate(dc['Genename']):
				print g
				y_arr = [[], [], []]
				ll = [[], [], []]
				a = [[], [], []]
				for r in xrange(dc['or'].shape[0]):
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
						av, _ = moving_average(prd, 26, ks=47)
						y_arr[oi].append(av)
						ll[oi].append(dc['L'][r])
						a[oi].append(dc['age'][r])
				for oi in xrange(3):
					y = np.array(y_arr[oi])
					L = np.array(ll[oi]) - Lav
					A = np.array(a[oi])
					if len(y.shape) < 2:
						print "Few samples!"
						continue
					print y.shape
					lx = np.linspace(np.min(L), np.max(L), 100)
					pca = PCA(n_components=2)
					pca.fit(y)
					print pca.explained_variance_ratio_
					yp = pca.transform(y)
					x = np.linspace(0,1,y.shape[1])
					plt.figure(figsize=plt.rcParams['figure.figsize'])
					plt.subplot(231)
					plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
					plt.subplot(232)
					m,b,r,p,s = stats.linregress(L, yp[:,0])
					plt.scatter(L, yp[:,0])
					plt.plot(lx, m*lx+b, color='r')
					plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
					plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
					plt.subplot(233)
					m,b,r,p,s = stats.linregress(L, yp[:,1])
					plt.scatter(L, yp[:,1])
					plt.plot(lx, m*lx+b, color='r')
					plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
					plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r,p))
					plt.subplot(234)
					plt.plot(x, y.T, alpha=0.4)
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
					plt.suptitle('{3} {0} a{1:.1f} n{2}'.format(g, np.average(dc['age']), y.shape[0], names[oi]))
					plt.savefig(ensure_dir("plots/mat/pca_{0}_{1}_{2}_{3}_{4}.pdf".format(li, dc['Flylinename'], int(nc), g, oi)))
					plt.clf()

	def length_binning(self, di=20):
		n_bins = 8
		vals = np.ones((n_bins, 2, 4, 2, 3))
		names = ['Symmetric', 'Ventral', 'Dorsal']
		for li in xrange(2):
			for dc in self.data_containers:
				nc = dc['nc'][0]
				Lav = dc['L'].mean()
				for i, g in enumerate(dc['Genename']):
					print g
					n, bins = np.histogram(dc['age'], bins=n_bins)
					idx = np.digitize(dc['age'], bins)
					sample_size = np.zeros((n_bins, 3))
					for bi in xrange(n_bins):
						pos = (idx == (bi+1))
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
								if li == 0:
									prd = pr
								else:
									prd = np.log(pr)
								if nc > 13:
									av, _ = moving_average(prd, 8)
								else:
									av, _ = moving_average(prd, 26, ks=47)
								y_arr[oi].append(av)
								ll[oi].append(dc['L'][r])
								a[oi].append(dc['age'][r])
						for oi in xrange(3):
							y = np.array(y_arr[oi])
							L = np.array(ll[oi])
							if len(y.shape) < 2:
								print "Few samples!"
								vals[bi,0,:,li,oi] = [1,0,0,0]
								vals[bi,1,:,li,oi] = [1,0,0,0]
								continue
							print y.shape
							n, bins = np.histogram(L, bins=4)
							idx_l = np.digitize(L, bins)
							pos_l = (idx_l == 1)
							r_list_l = np.where(pos_l)[0]
							lower_y = y[r_list_l]
							print lower_y.shape, L[r_list_l]
							pos_u = (idx_l == 4)
							r_list_u = np.where(pos_u)[0]
							upper_y = y[r_list_u]
							print upper_y.shape, L[r_list_u]
							if upper_y.shape[0] < 2 or lower_y.shape[0] < 2:
								continue
							'''ks 2 sample'''
							pval=[]
							for k in xrange(lower_y.shape[1]):
								try:
									_, p = stats.ks_2samp(lower_y[:,k], upper_y[:,k])
								except ValueError:
									p = 0
								pval.append(p)
							''' chi square'''
							av = np.mean(y, axis=0)
							var = np.var(y, axis=0)
							lower_p = []
							upper_p = []
							for k in xrange(lower_y.shape[1]):
								ch_lower = np.sum(np.power(av[k]-lower_y[:,k],2))/var[k]
								ch_upper = np.sum(np.power(av[k]-upper_y[:,k],2))/var[k]
								df_lower = (lower_y.shape[0]-1)
								df_upper = (upper_y.shape[0]-1)
								lower_p.append(1-stats.chi2.cdf(ch_lower, df_lower))
								upper_p.append(1-stats.chi2.cdf(ch_upper, df_upper))
							plt.subplot(321)
							plt.plot(lower_y.T)
							plt.subplot(322)
							plt.plot(upper_y.T)
							plt.subplot(323)
							x = np.linspace(0,1,lower_y.shape[1])
							lyav = np.mean(lower_y, axis=0)
							lystd = np.std(lower_y, axis=0)
							plt.plot(x,lyav)
							plt.fill_between(x,lyav-lystd, lyav+lystd, alpha=0.3)
							plt.subplot(324)
							x = np.linspace(0,1,upper_y.shape[1])
							lyav = np.mean(upper_y, axis=0)
							lystd = np.std(upper_y, axis=0)
							plt.plot(x,lyav)
							plt.fill_between(x,lyav-lystd, lyav+lystd, alpha=0.3)
							plt.subplot(325)
							plt.plot(x, pval)
							plt.plot(x, np.ones(shape=x.shape)*0.05, "g--", label=r"5\% confidence")
							plt.plot(x, np.ones(shape=x.shape)*0.01, "r--", label=r"1\% confidence")
							plt.yscale('log')
							plt.ylabel('pvalue', fontsize=10)
							leg = plt.legend(loc="lower left", fontsize = 8)
							leg.get_frame().set_alpha(0.5)
							plt.subplot(326)
							plt.plot(x, lower_p, label='lower L')
							plt.plot(x, upper_p, label='upper L')
							plt.plot(x, np.ones(shape=x.shape)*0.05, "g--", label=r"5\% confidence")
							plt.plot(x, np.ones(shape=x.shape)*0.01, "r--", label=r"1\% confidence")
							plt.yscale('log')
							plt.ylabel('pvalue', fontsize=10)
							leg = plt.legend(loc="lower left", fontsize = 8)
							leg.get_frame().set_alpha(0.5)
							plt.suptitle('{1} {0}'.format(g, names[oi]))
							plt.savefig(ensure_dir("plots/tbin_{0}_{1}_{2}_{3}_{4}.pdf".format(li, int(nc), g, oi, bi)))
							plt.clf()

	def slope(self, di=20):
		for dc in self.data_containers:
			nc = dc['nc'][0]
			Lav = dc['L'].mean()
			for i, g in enumerate(dc['Genename']):
				print g
				n_bins = 3
				n, bins = np.histogram(dc['age'], bins=n_bins)
				idx = np.digitize(dc['age'], bins)
				vals = np.empty((n_bins, 2, 4))
				sample_size = np.empty((n_bins))
				for bi in xrange(n_bins):
					pos = (idx == (bi+1))
					r_list = np.where(pos)[0]
					y = []
					ll = []
					a = []
					for r in r_list:
						for e in xrange(2):
							#if dc['or'][r] == 1 or (dc['or'][r] == 1 and e == 2):
							#	'''skip dorsal measurements'''
							#	continue
							if di > 0:
								pr = dc['meas'][di:-di,i,e,r]
							else:
								pr = dc['meas'][:,i,e,r]
							nans, xp = nan_helper(pr)
							pr[nans] = np.interp(xp(nans), xp(~nans), pr[~nans])
							if nc > 13:
								av, _ = moving_average(np.log(pr), 8)
							else:
								av, _ = moving_average(np.log(pr), 26, ks=47)
							y.append(av)
							ll.append(dc['L'][r])
							a.append(dc['age'][r])
					y = np.array(y)
					L = np.array(ll) - Lav
					sample_size[bi] = y.shape[0]
					if y.shape[0] < 10:
						print "Few samples!"
						continue
					print y.shape
					lx = np.linspace(np.min(L), np.max(L), 100)
					params = np.empty((5, y.shape[1]))
					for k in xrange(y.shape[1]):
						params[:, k] = stats.linregress(L, y[:,k])
					x = np.linspace(0,1,y.shape[1])
					plt.subplot(231)
					plt.plot(x, y.T, alpha=0.4)
					plt.title('data')
					plt.subplot(232)
					plt.plot(x, params[2], alpha=0.5)
					plt.title('r')
					plt.subplot(233)
					plt.plot(x, params[3], alpha=0.5)
					plt.title('p')
					plt.plot(x,np.ones(shape=x.shape[0])*0.05, "g--", label=r"5\% confidence")
					plt.plot(x,np.ones(shape=x.shape[0])*0.01, "r--", label=r"1\% confidence")
					plt.title('pval')
					plt.yscale('log')
					leg = plt.legend(loc="lower left", fontsize = 8)
					leg.get_frame().set_alpha(0.5)
					plt.subplot(234)
					plt.fill_between(x, params[0]-2*params[4], params[0]+2*params[4], alpha=0.3)
					plt.plot(x, params[0])
					plt.title('slope and 95CI')
					plt.suptitle('{0} a{1:.2f} n{2}'.format(g, np.average(dc['age'][pos]), y.shape[0]))
					plt.savefig(ensure_dir("plots/slope_{0}_{1}_{2}.pdf".format(int(nc), g, bi)))
					plt.clf()

	def mv_pca(self, di=20):
		for dc in self.data_containers:
			nc = dc['nc'][0]
			Lav = dc['L'].mean()
			n_bins = 8
			n, bins = np.histogram(dc['age'], bins=n_bins)
			idx = np.digitize(dc['age'], bins)
			vals = np.empty((n_bins, 2, 4))
			sample_size = np.empty((n_bins))
			for bi in xrange(n_bins):
				pos = (idx == (bi+1))
				r_list = np.where(pos)[0]
				y = []
				ll = []
				for r in r_list:
					for e in xrange(2):
						#if dc['or'][r] == 2 or (dc['or'][r] == 1 and e == 1):
						#	'''skip dorsal measurements'''
						#	continue
						tav = []
						for i,_ in enumerate(dc['Genename']):
							if di > 0:
								pr = dc['meas'][di:-di,i,e,r]
							else:
								pr = dc['meas'][:,i,e,r]
							nans, xp = nan_helper(pr)
							pr[nans] = np.interp(xp(nans), xp(~nans), pr[~nans])
							if nc > 13:
								av, _ = moving_average(np.log(pr), 8)
							else:
								av, _ = moving_average(np.log(pr), 26, ks=47)
							tav.append(av)
						y.append(np.array(tav).flatten())
						ll.append(dc['L'][r])
				y = np.array(y)
				L = np.array(ll) - Lav
				sample_size[bi] = y.shape[0]
				if y.shape[0] < 10:
					print "Few samples!"
					continue
				print y.shape
				lx = np.linspace(np.min(L), np.max(L), 100)
				pca = PCA(n_components=2)
				pca.fit(y)
				print pca.explained_variance_ratio_
				yp = pca.transform(y)
				x = np.linspace(0,1,y.shape[1])
				plt.subplot(231)
				plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
				plt.subplot(232)
				m,b,r,p,s = stats.linregress(L, yp[:,0])
				vals[bi,0, :] = [p,m,r,s]
				plt.scatter(L, yp[:,0])
				plt.plot(lx, m*lx+b, color='r')
				plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
				plt.title("pc1 r:{0:.2f},p:{1:.2f}".format(r,p))
				plt.subplot(233)
				m,b,r,p,s = stats.linregress(L, yp[:,1])
				vals[bi,1] = [p,m,r,s]
				plt.scatter(L, yp[:,1])
				plt.plot(lx, m*lx+b, color='r')
				plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
				plt.title("pc2 r:{0:.2f},p:{1:.2f}".format(r,p))
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
				plt.suptitle('a{0:.2f} n{1}'.format(np.average(dc['age'][pos]), y.shape[0]))
				plt.savefig(ensure_dir("plots/mv_pca_{0}_{1}.pdf".format(int(nc), bi)))
				plt.clf()
				plt.subplot(411)
				plt.plot(vals[:,0,0], label='1st pc')
				plt.plot(vals[:,1,0], label='2nd pc')
				plt.plot(np.ones(shape=vals.shape[0])*0.05, "g--", label=r"5\% confidence")
				plt.plot(np.ones(shape=vals.shape[0])*0.01, "r--", label=r"1\% confidence")
				plt.title('pval')
				plt.yscale('log')
				leg = plt.legend(loc="lower left", fontsize = 8)
				leg.get_frame().set_alpha(0.5)
				plt.subplot(412)
				plt.errorbar(np.arange(vals.shape[0]), vals[:,0,1], yerr=vals[:,0,3], label='1st pc')
				plt.errorbar(np.arange(vals.shape[0]), vals[:,1,1], yerr=vals[:,1,3], label='2nd pc')
				plt.plot(np.zeros(shape=vals.shape[0]), "r--", label=r"no slope")
				plt.title('slope')
				leg = plt.legend(loc="lower left", fontsize = 8)
				leg.get_frame().set_alpha(0.5)
				plt.subplot(413)
				plt.plot(vals[:,0,2], label='1st pc')
				plt.plot(vals[:,1,2], label='2nd pc')
				plt.title('r')
				leg = plt.legend(loc="lower left", fontsize = 8)
				leg.get_frame().set_alpha(0.5)
				plt.subplot(414)
				plt.plot(sample_size)
				plt.savefig(ensure_dir("plots/mv_tpca_{0}.pdf".format(int(nc))))
				plt.clf()

if __name__ == '__main__':
	sc = ScalingContainer()
	#sc.length_binning()
	sc.pca_gregor()
	#sc.pca_maternal()