__author__ = 'tiago'
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
import scipy.stats as stats
import itertools

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

def stretch(series, factor):
	x = np.linspace(0,1,series.shape[0])
	y = UnivariateSpline(x, series, s=0)
	return y(x*factor)

n_samples = 60
n_ap = 1000
L = np.random.normal(loc=450, scale=50, size=(n_samples))
lr = lambda l:0.5*l/450+0.5
ap = np.linspace(0,1,n_ap)

for var in itertools.product(['gt_var', 'gt', 'bcd'], [True, False], [True, False], [True, False]):
	gt = var[0]
	vt = var[1]
	vs = var[2]
	s_corr = var[3]

	if gt == 'gt_var':
		y = np.empty((n_samples, n_ap))
		for i in xrange(n_samples):
			sv = np.random.normal(loc=0.05, scale=0.01)
			if s_corr:
				sv *= lr(L[i])
			y[i] = 0.2*stats.norm.pdf(ap, loc=0.5, scale=sv) + 0.5
	else:
		if gt == 'gt':
			av = 0.2*stats.norm.pdf(ap, loc=0.5, scale=0.05) + \
	             0.01*stats.norm.pdf(ap, loc=0.2, scale=0.02) + 0.5
		else:
			av = 1-ap/1.2
		y = np.tile(av, n_samples).reshape(n_samples, n_ap)

	for i in xrange(n_samples):
		if vt:
			y[i] += np.random.normal(loc=1, scale=0.1)
		if vs:
			y[i] = stretch(y[i], lr(L[i]))

	y += np.random.normal(scale=0.05, size=(n_samples, n_ap))

	for i in xrange(n_samples):
		y[i], _ = moving_average(y[i], 26)

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
	plt.scatter(L, yp[:,0])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
	plt.title("pc1 r:{0:.2f},p:{1:.2f}".format(r,p))
	plt.subplot(233)
	m,b,r,p,s = stats.linregress(L, yp[:,1])
	plt.scatter(L, yp[:,1])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.1, color='r')
	plt.title("pc2 r:{0:.2f},p:{1:.2f}".format(r,p))
	plt.subplot(234)
	plt.plot(x, y.T, alpha=0.5)
	plt.title('data')
	plt.subplot(235)
	s = np.random.normal(scale=np.std(yp[:,0]), size=n_samples)
	v = np.vstack([s, np.zeros(n_samples)]).T
	yt = pca.inverse_transform(v)
	plt.plot(x, yt.T, alpha=0.5)
	plt.title('pc1')
	plt.subplot(236)
	s = np.random.normal(scale=np.std(yp[:,1]), size=n_samples)
	v = np.vstack([np.zeros(n_samples), s]).T
	yt = pca.inverse_transform(v)
	plt.plot(x, yt.T, alpha=0.5)
	plt.title('pc2')
	plt.savefig(ensure_dir("plots/test/pca_test_{0}_{1}_{2}_{3}.pdf".format(gt, vt, vs, s_corr)))
	plt.clf()