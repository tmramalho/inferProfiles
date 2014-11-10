__author__ = 'tiago'


import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt

avL = 500
stdL = 30
stdM = 0.1
m = -1/50.0
b = 10

def sample_conc_from_positions(nx):
	x = np.random.rand(nx)
	l = np.random.normal(loc=avL, scale=stdL, size=nx)
	real_pos = x*l
	conc = m*real_pos+b + np.random.normal(scale=stdM, size=nx)
	return np.vstack([x, conc])

def sample_2d_conc_from_positions(nx):
	x = np.random.rand(nx)
	l = np.random.normal(loc=avL, scale=stdL, size=nx)
	real_pos = x*l
	conc_1 = m*real_pos+b + np.random.normal(scale=stdM, size=nx)
	conc_2 = m*(l-real_pos)+b + np.random.normal(scale=stdM, size=nx)
	return np.vstack([x, conc_1, conc_2])

values = sample_conc_from_positions(1e6)
'''
Get optimal estimator
'''
n_bins = 100
bins = np.linspace(np.min(values[1]), np.max(values[1]), n_bins+1)
inds = np.digitize(values[1], bins)
est = []
for i in xrange(n_bins):
	px = values[0, np.where(inds==(i+1))]
	mu = (bins[i+1]+bins[i])/2
	est.append((mu, np.mean(px)))
est = np.array(est)
mask = np.isnan(est[:,1])
s = ip.UnivariateSpline(est[~mask,0], est[~mask,1], s=0)
plt.plot(bins, s(bins))
plt.savefig('theory/estimator1d.pdf')
plt.clf()

'''
Get error from estimator
'''
bins = np.linspace(0,1, n_bins+1)
inds = np.digitize(values[0], bins)
err = []
for i in xrange(n_bins):
	pc = values[1, np.where(inds==(i+1))].flatten()
	x_std = np.std(s(pc))
	x = (bins[i]+bins[i+1])/2
	err.append((x, x_std))
err = np.array(err)
plt.plot(err[:,0], err[:,1])
plt.savefig('theory/error1d.pdf')
plt.clf()

values = sample_2d_conc_from_positions(1e6)

'''
Get optimal estimator
'''
n_bins = 100
binsx = np.linspace(np.min(values[1]), np.max(values[1]), n_bins+1)
binsy = np.linspace(np.min(values[2]), np.max(values[2]), n_bins+1)
indsx = np.digitize(values[1], binsx)
indsy = np.digitize(values[2], binsy)
try:
	est = np.load('theory/2d_estimates.npy')
except IOError:
	print 'gotta generate the estimator array, grab some coffee...'
	est = np.zeros((3,n_bins,n_bins))
	for i in xrange(n_bins):
		for j in xrange(n_bins):
			px = values[0, np.where((indsx==(i+1)) & (indsy==(j+1)))]
			est[0,i,j] = np.mean(px)
			est[1,i,j] = (binsx[i]+binsx[i+1])/2
			est[2,i,j] = (binsy[j]+binsy[j+1])/2
	np.save('theory/2d_estimates.npy', est)
z = est[0].flatten()
x = est[1].flatten()
y = est[2].flatten()
plt.imshow(est[0],
           extent=[x.min(), x.max(), y.min(), y.max()],
           interpolation='bilinear', origin='lower', aspect='auto')
plt.savefig('theory/estimator2d_raw.pdf')
plt.clf()
mask = np.isnan(z)
rbfi = ip.Rbf(x[~mask], y[~mask], z[~mask], function='linear')
xx, yy = np.meshgrid(binsx, binsy)
zz = rbfi(xx.flatten(), yy.flatten())
plt.imshow(zz.reshape(xx.shape), vmin=0, vmax=1,
           extent=[binsx.min(), binsx.max(), binsy.min(), binsy.max()],
           interpolation='bilinear', origin='lower')
plt.savefig('theory/estimator2d.pdf')
plt.clf()

'''
Get error from estimator
'''
bins = np.linspace(0,1, n_bins+1)
inds = np.digitize(values[0], bins)
err = []
for i in xrange(n_bins):
	print i
	pc1 = values[1, np.where(inds==(i+1))].flatten()
	pc2 = values[2, np.where(inds==(i+1))].flatten()
	x_std = np.std(rbfi(pc1, pc2))
	x = (bins[i]+bins[i+1])/2
	err.append((x, x_std))
err = np.array(err)
plt.plot(err[:,0], err[:,1])
plt.savefig('theory/error2d.pdf')