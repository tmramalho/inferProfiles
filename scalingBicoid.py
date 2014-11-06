
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

if __name__ == '__main__':
	#rc('text', usetex=True)
	#rc('font',**{'family':'serif','serif':['Computer Modern']})
	fly = ldp.LiuHugeDataset()
	# lower_bins, lower_av, lower_std = fly.get_length_percentile_binned_data(20)
	# upper_bins, upper_av, upper_std = fly.get_length_percentile_binned_data(20, lower=False)
	#
	# pval = []
	# for i in xrange(len(lower_bins)):
	# 	try:
	# 		_, p = stats.ks_2samp(lower_bins[i], upper_bins[i])
	# 	except ValueError:
	# 		p = 0
	# 	pval.append(p)
	#
	# plt.subplot(211)
	# plt.errorbar(np.linspace(0,1,len(lower_av)), lower_av, yerr=lower_std)
	# plt.errorbar(np.linspace(0,1,len(upper_av)), upper_av, yerr=upper_std)
	# plt.yscale('log')
	# plt.grid(False, which="minor")
	# plt.subplot(212)
	# px = np.linspace(0,1,len(pval))
	# plt.plot(px, pval)
	# plt.plot(px, np.ones(shape=px.shape)*0.05, "g--", label=r"5\% confidence")
	# plt.plot(px, np.ones(shape=px.shape)*0.01, "r--", label=r"1\% confidence")
	# plt.yscale('log')
	# plt.ylabel('pvalue', fontsize=10)
	# leg = plt.legend(loc="lower left", fontsize = 8)
	# leg.get_frame().set_alpha(0.5)
	# plt.savefig("results/a_scaling.pdf")
	# plt.clf()

	# profiles, _ = fly.getIndividualProfiles(100, 1)
	# int_noise = []
	# for i,p in enumerate(profiles):
	# 	y = np.log(p[1])
	# 	av, va = moving_average(y, 5, 15)
	# 	plt.scatter(p[0], y)
	# 	plt.plot(p[0], av)
	# 	plt.fill_between(p[0], av-np.sqrt(va), av+np.sqrt(va), facecolor='r', alpha = 0.4)
	# 	plt.savefig("results/bcd_{0}.pdf".format(i))
	# 	plt.clf()
	# 	int_noise.append(va)
	# np.save("results/va.npy", int_noise)
	profiles, lens = fly.getIndividualProfiles(300, 1)
	L = lens[:, 0]-np.mean(lens[:, 0])
	lx = np.linspace(np.min(L), np.max(L), 100)
	print L.shape
	pr = []
	for i,p in enumerate(profiles):
		profile = UnivariateSpline(p[0], np.log(p[1]), s=1000)
		x = np.linspace(0,0.9,90)
		pr.append(profile(x))
	y = np.array(pr)
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
	plt.savefig(ensure_dir("plots/bcdpca.pdf"))
	plt.clf()