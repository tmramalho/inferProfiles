
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

def do_pca_analysis(profiles, lens, name=''):
	L = lens-np.mean(lens)
	lx = np.linspace(np.min(L), np.max(L), 100)
	#print L.shape
	pr = []
	for i,p in enumerate(profiles):
		profile = UnivariateSpline(p[0], np.log(p[1]), s=1000)
		x = np.linspace(0,0.9,90)
		pr.append(profile(x))
	y = np.array(pr)
	pca = PCA(n_components=2)
	pca.fit(y)
	#print pca.explained_variance_ratio_
	yp = pca.transform(y)
	x = np.linspace(0,0.9,y.shape[1])
	plt.subplot(231)
	plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
	plt.subplot(232)
	m,b,r1,p1,s = stats.linregress(L, yp[:,0])
	plt.scatter(L, yp[:,0])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r1,p1))
	plt.subplot(233)
	m,b,r2,p2,s = stats.linregress(L, yp[:,1])
	plt.scatter(L, yp[:,1])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r2,p2))
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
	plt.suptitle('session {0} n:{1}'.format(name, L.shape[0]))
	plt.savefig(ensure_dir("plots/2XA/pca_{0}.pdf".format(name)))
	plt.clf()
	if L.shape[0] > 11:
		print name, L.shape[0], p1, r1, p2, r2, np.std(L)

if __name__ == '__main__':
	bcd_data = sio.loadmat('data/scaling_data/DataSets/2XAAllEmbryos.mat', squeeze_me=True)
	bcd_data = bcd_data['AllEmbryos_2XA']
	sessions = np.unique(bcd_data['Session'])
	quality = 2
	for s in sessions:
		targ_embryos = np.where((bcd_data['Session'] == s) & (bcd_data['Quality'] <= quality))
		full_profiles = bcd_data['Gradient'][targ_embryos]
		lengths = bcd_data['EggLength'][targ_embryos].astype('float64')
		left_profiles = []
		right_profiles = []
		for p in full_profiles:
			left_profiles.append(np.transpose(p['left'].item()))
			right_profiles.append(np.transpose(p['right'].item()))
		do_pca_analysis(left_profiles, lengths, '{0}_{1}_left'.format(s,quality-1))
		do_pca_analysis(right_profiles, lengths, '{0}_{1}_right'.format(s,quality-1))
		do_pca_analysis(right_profiles + left_profiles,
		                np.hstack((lengths, lengths)), '{0}_{1}_both'.format(s,quality-1))
