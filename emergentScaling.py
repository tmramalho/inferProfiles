__author__ = 'tiago'

import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt

avL = 500
stdL = 30
stdM = 0.1
m = -1/50.0
b = 10

def sample_conc_given_l(x, l):
	real_pos = np.outer(x,l)
	conc = m*real_pos+b + np.random.normal(scale=stdM, size=real_pos.shape)
	return conc

'''
Evolve the system in time
'''
n_pos = 100
n_samples = 1000
n_times = 1000
a = 0.5
mu_a = 0.01
y = 0.5*b*np.ones((n_pos, n_samples))
l = np.random.normal(loc=avL, scale=stdL, size=n_samples)
x = np.linspace(0, 1, n_pos)

for t in xrange(n_times):
	mu_sample = sample_conc_given_l(x, l)
	w = np.pad(y, ((1,1),(0,0)), mode='constant', constant_values=(b,0))
	y_est = mu_a*mu_sample + (1-mu_a)/2*(w[:-2] + w[2:])
	y = a*y + (1-a)*y_est + np.random.normal(scale=stdM, size=y.shape)

mu_samp = sample_conc_given_l(x, l)
plt.subplot(211)
plt.plot(mu_samp[:,::100])
plt.ylim([0,b])
plt.subplot(212)
plt.plot(y[:,::100])
plt.ylim([0,b])
plt.savefig('theory/final.pdf')
plt.clf()

plt.hist(mu_samp[-1], bins=20)
plt.hist(y[-1], bins=20)
plt.savefig('theory/last_hist.pdf')
plt.clf()

conc = y.flatten()
x_val = np.repeat(x, n_samples)
values = np.vstack([x_val, conc])
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
plt.savefig('theory/emergent_estimator1d.pdf')
plt.clf()

'''
Get error from estimator
'''
err = []
for i in xrange(n_pos):
	pc = values[1, i*n_pos:(i+1)*n_pos].flatten()
	x_std = np.std(s(pc))
	x = i/float(n_pos)
	err.append((x, x_std))
err = np.array(err)
plt.plot(err[:,0], err[:,1])
plt.savefig('theory/emergent_error1d.pdf')
plt.clf()