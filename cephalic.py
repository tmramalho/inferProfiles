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

feng_sdata = sio.loadmat('data/Feng_Bcd_GFP_TwoPhoton_2012.mat', squeeze_me=True)
cf_data = []
L_data = []
jmax = feng_sdata['BCDEm']['M'].item()['Em'].shape[0]
col = []

for j,em in enumerate(feng_sdata['BCDEm']['M'].item()['Em']):
	try:
		ll = em['Prop'].shape[0]
	except IndexError:
		continue
	cf_temp = []
	L_temp = []
	for i in range(ll):
		if not np.isnan(em['CF'][i]) and em['ID'][0]['FlyLine0'] == 'BCD20A' and em['Prop'][i]['CFflag'] == 1: #em['Prop'][i]['Egglength'] > 400:
			cf_temp.append(em['CF'][i])
			L_temp.append(em['Prop'][i]['Egglength'])
			col.append(j/(jmax-1))
	if len(cf_temp) < 6:
		continue
	cf_data.append(cf_temp)
	L_data.append(L_temp)

cf = np.concatenate(cf_data)
L = np.concatenate(L_data)
hsv = plt.get_cmap('jet')
lx = np.linspace(np.min(L), np.max(L), 100)
m,b,r,p,s = stats.linregress(L, cf)
plt.plot(lx, m*lx+b, color='r')
plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
plt.scatter(L, cf, color=hsv(col))
plt.title('p{0:0.2e} r{1:0.2f}'.format(p, r*r))
plt.xlabel('embryo length in microns')
plt.ylabel('cf position in percent EL')
plt.savefig('plots/cf/cf.pdf')
plt.clf()

r_values = []
p_values = []
s_values = []
for i, (c, l) in enumerate(zip(cf_data, L_data)):
	c = np.array(c)
	l = np.array(l)
	m,b,r,p,s = stats.linregress(l, c)
	r_values.append(r*r)
	p_values.append(p)
	s_values.append(c.shape[0])
	plt.scatter(l, c)
	lx = np.linspace(np.min(l), np.max(l), 100)
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title('p{0:0.2e} r{1:0.2f}'.format(p, r*r))
	plt.xlabel('embryo length in microns')
	plt.ylabel('cf position in percent EL')
	plt.savefig('plots/cf/cf{0}_{1}.pdf'.format(i,int(r*r*100)))
	plt.clf()

plt.yscale('log')
plt.scatter(r_values, p_values, s=s_values)
plt.title('regression results for each individual session')
plt.ylabel('regression p value')
plt.xlabel('r^2')
plt.savefig('plots/cf/cf_ind.pdf')