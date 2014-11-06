
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
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		av, va = moving_average(np.log(p), 46, 100)
		profiles.append(av)
	y = np.array(pr)
	pca = PCA(n_components=2)
	pca.fit(y)
	#print pca.explained_variance_ratio_
	yp = pca.transform(y)
	x = np.linspace(0,0.9,y.shape[1])
	plt.subplot(231)
	plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
	plt.subplot(232)
	m,b,r,p,s = stats.linregress(L, yp[:,0])
	min_p = p
	min_r = r
	plt.scatter(L, yp[:,0])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(233)
	m,b,r,p,s = stats.linregress(L, yp[:,1])
	if p < min_p:
		min_p = p
		min_r = r
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
	plt.suptitle('{0} n:{1}'.format(name, L.shape[0]))
	plt.savefig(ensure_dir("plots/full_mutant/pca_{0}.pdf".format(name)))
	plt.clf()
	print name, L.shape[0], min_p, min_r

if __name__ == '__main__':
	for csv_filename in os.listdir('data/mutant'):
		if csv_filename.endswith(".mat"):
			print csv_filename
			gap_data = sio.loadmat('data/mutant/{0}'.format(csv_filename), squeeze_me=True)
			gap_data = gap_data['data']
			pos = (gap_data['age'] >= 145) & (gap_data['age'] <= 155)
			ind = np.where(pos)[0]
			do_pca_analysis(gap_data['Kni'][ind], gap_data['L'][ind], 'Kni early {0}'.format(csv_filename[:-4]))
			do_pca_analysis(gap_data['Kr'][ind], gap_data['L'][ind], 'Kr early {0}'.format(csv_filename[:-4]))
			do_pca_analysis(gap_data['Hb'][ind], gap_data['L'][ind], 'Hb early {0}'.format(csv_filename[:-4]))
			do_pca_analysis(gap_data['Gt'][ind], gap_data['L'][ind], 'Gt early {0}'.format(csv_filename[:-4]))
			pos = (gap_data['age'] >= 170) & (gap_data['age'] <= 180)
			do_pca_analysis(gap_data['Kni'][ind], gap_data['L'][ind], 'Kni late {0}'.format(csv_filename[:-4]))
			do_pca_analysis(gap_data['Kr'][ind], gap_data['L'][ind], 'Kr late {0}'.format(csv_filename[:-4]))
			do_pca_analysis(gap_data['Hb'][ind], gap_data['L'][ind], 'Hb late {0}'.format(csv_filename[:-4]))
			do_pca_analysis(gap_data['Gt'][ind], gap_data['L'][ind], 'Gt late {0}'.format(csv_filename[:-4]))
