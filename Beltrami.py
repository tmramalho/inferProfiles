'''
Created on Dec 20, 2013

@author: tiago
'''

import numpy as np
import time
import random
from scipy.interpolate import UnivariateSpline
from LiuDatasetProcess import LiuDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == '__main__':
	x = np.linspace(0, 1, 1000)
	plt.rcParams['axes.grid']=False
	plt.rcParams['axes.linewidth']=2
	plt.rcParams['xtick.major.size']=4
	plt.rcParams['ytick.major.size']=4
	plt.rcParams['xtick.major.width']=2
	plt.rcParams['ytick.major.width']=2
	plt.rcParams['axes.edgecolor']='555555'
	plt.rcParams['xtick.labelsize']=20
	plt.rcParams['ytick.labelsize']=20
	plt.rcParams['text.usetex']=True
	plt.rcParams['axes.labelcolor']='333333'
	plt.rcParams['xtick.color']='333333'
	plt.rcParams['ytick.color']='333333'
	plt.rcParams['font.size']=20
	nu = 0.01
	s = 1
	plt.figure(figsize=(10,4))
	plt.subplot(121)
	plt.plot(x, (nu - 1)*x + 1, lw=2, ls='--')
	plt.plot(x, np.power((np.sqrt(nu) - 1)*x + 1,2), lw=2, ls=':')
	plt.plot(x, np.exp(x*np.log(nu)), lw=2)
	plt.xlabel(r'$x/L$')
	plt.ylabel(r'$\mu(x/L)$')
	plt.subplot(122)
	plt.plot(x, 3*s/np.power(1-x,2), lw=2, ls='--')
	plt.plot(x, 3*s/np.power(2*(np.sqrt(x)-1),2), lw=2, ls=':')
	plt.plot(x, 6*s/(np.power(np.log(x),2)*(2+s)), lw=2)
	plt.yscale('log')
	plt.xlabel(r'$\nu$')
	plt.ylabel(r'$C(\nu)$')
	plt.tight_layout()
	plt.show()