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

print 'Reading matlab...'
bcd_data = sio.loadmat('data/scaling_data/DataSets/ScalingDataTempVariedBcd.mat', squeeze_me=True)
dat = bcd_data['RawData'].item()[0]['Em'].item()
print 'Generating data...'
L = dat['EL'].astype('float64')
L -= np.mean(L)
profiles = []
print L.shape
for p in dat['Profile']:
	mask = np.isnan(p)
	p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
	av, va = moving_average(np.log(p), 46, 100)
	profiles.append(av)
lx = np.linspace(np.min(L), np.max(L), 100)
y = np.array(profiles)
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
plt.savefig(ensure_dir("plots/bcdpca_TempVaried.pdf"))
plt.clf()