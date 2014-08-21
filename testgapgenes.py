'''
Created on Mar 11, 2013

@author: tiago
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fminbound
from scipy.signal import filtfilt, butter
from scipy.optimize import leastsq
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as Fig
import matplotlib.backends.backend_agg as bkc
import matplotlib.gridspec as gridspec
import os

sigma = lambda x, p: p[0] * (p[1]*x*x + x)

class ComboPlot(object):
	
	def __init__(self):
		self.fig = Fig.Figure()
		self.inc = 1
		self.ax = []
		self.gs = gridspec.GridSpec(4, 2, width_ratios=[15,1])
		
	def plot(self, x, y, name):
		if self.inc == 1:
			ax = self.fig.add_subplot(self.gs[self.inc - 1, 0])
		else:
			ax = self.fig.add_subplot(self.gs[self.inc - 1, 0], sharex=self.ax1)
		hm = np.array([y])
		self.im = ax.imshow(hm, interpolation = 'bicubic', cmap=plt.cm.Blues_r, aspect='auto', vmin = 0, vmax = 0.002)
		ax.set_ylabel(name)
		plt.setp(ax.get_yticklabels(), visible=False)
		if self.inc == 1:
			self.ax1 = ax #save for sharing
		if self.inc < 4:
			plt.setp(ax.get_xticklabels(), visible=False)
		ax.set_xticks((0,200,400,600,800,1000))
		ax.set_xticklabels((0, 0.2, 0.4, 0.6, 0.8, 1))
		self.inc += 1
	
	def save(self):
		ax = self.fig.add_subplot(self.gs[:, 1])
		self.fig.colorbar(self.im, cax=ax)
		self.fig.subplots_adjust(hspace=0)
		basecanv = bkc.FigureCanvasAgg(self.fig)
		basecanv.print_figure("gap/allCombo.pdf",format="pdf",bbox_inches='tight',pad_inches=0.1)
		
def plotFisherInfo(mprof, sprof, sname, combo):
	t = np.linspace(0, 1, 1000)
	
	minf = lambda x: -1 * mprof(x)
	minx = fminbound(minf, 0, 1)
	fval = mprof(minx)
	plt.subplot(321)
	plt.title("mu")
	plt.plot(t, mprof(t)/fval)
	plt.subplot(323)
	plt.title("dmu/dx")
	plt.plot(t, mprof(t,1)/fval)
	plt.subplot(322)
	plt.title("var")
	plt.plot(t, sprof(t)/fval/fval)
	plt.subplot(324)
	plt.title("dvar/dx")
	plt.plot(t, sprof(t, 1)/fval/fval)
	plt.subplot(325)
	plt.title("dmu/dx^2")
	plt.yscale("log")
	plt.plot(t, np.power(mprof(t,1)/fval, 2))
	plt.subplot(326)
	plt.title("dvar/dx^2")
	plt.yscale("log")
	plt.plot(t, np.power(sprof(t, 1)/fval/fval,2))
	plt.savefig("gap/derivGap"+sname+".pdf")
	plt.clf()
	
	fi = lambda a, sa, sp: 2*np.power(sa, 2)/ (np.power(a,2)*2*sa+np.power(sp,2))
	fiapp = lambda a, sa, sp: sa / (np.power(a,2))
	plt.xlim(0, 1)
	#plt.ylim(1e-5, 10)
	plt.yscale("log")
	print 'whop whop'
	plt.plot(t, fi(mprof(t,1)/fval, sprof(t)/fval/fval, sprof(t, 1)/fval/fval))
	plt.plot(t, fiapp(mprof(t,1)/fval, sprof(t)/fval/fval, sprof(t, 1)/fval/fval), 'r')
	plt.legend(('Full FI', '1st order FI'),
	   'upper right')
	plt.savefig("gap/fisherInfoGap"+sname+".pdf")
	plt.clf()
	
	combo.plot(t, fi(mprof(t,1)/fval, sprof(t)/fval/fval, sprof(t, 1)/fval/fval), sname)
		
def plotFisherInfoParam(mprof, sprof, sname, plsq):
	t = np.linspace(0, 1, 1000)
	
	minf = lambda x: -1 * mprof(x)
	minx = fminbound(minf, 0, 1)
	fval = mprof(minx)
	s = plsq[0]/fval
	se = 0
	p = plsq[1]*fval
	print fval, s, p
	fi = lambda j, jp: (np.power(jp, 2) * (s + 2 * j * (1 + p * j) * (1 + 2 * p * s)))/(2 * np.power(j, 2) * np.power(1 + p * j, 2) * s)
						
	fiapp = lambda j, jp: (np.power(jp, 2))/(4 * (j + p * j * j + se) * s)
	plt.xlim(0, 1)
	plt.yscale("log")
	plt.plot(t, fi(mprof(t)/fval, mprof(t,1)/fval))
	plt.plot(t, fiapp(mprof(t)/fval, mprof(t,1)/fval), 'r')
	plt.legend(('Full FI', '1st order FI'),
	   'upper right')
	plt.savefig("gap/fisherInfoParamGap"+sname+".pdf")
	plt.clf()
		
def fitNoiseExpression(x, y, sname):
	p0 = [10, 4]
	plsq = leastsq(residuals, p0, args=(y, x))
	plt.plot(x, y, marker='.', ls='None')
	print sname, plsq[0]
	xp = np.sort(x)
	plt.plot(xp, sigma(xp, plsq[0]))
	plt.savefig("gap/exp"+sname+".pdf")
	plt.clf()
	return plsq[0]

def residuals(p, y, x):
	err = y - sigma(x, p)
	return err

if __name__ == '__main__':
	filenames = ["data/Raw_Profiles/Data_Hb_raw.csv","data/Raw_Profiles/Data_Gt_raw.csv",
			"data/Raw_Profiles/Data_Kni_raw.csv","data/Raw_Profiles/Data_Kr_raw.csv"]
	combo = ComboPlot()
	for fn in filenames:
		sname = os.path.basename(fn)
		sname = sname[:-3]
		data = pd.read_csv(fn, quotechar="\"", delimiter=",", skiprows=[0,2], header=0)
		average = data.mean()
		average = average[4:].fillna(0).values
		std = data.var()
		std = std[4:].fillna(0).values
		pos = np.linspace(0, 1, average.shape[0])
		b, a = butter(4, 0.03)
		avfilt = filtfilt(b, a, average)
		avint = UnivariateSpline(pos, avfilt, s=None, k=4)
		plt.subplot(211)
		plt.plot(pos, average)
		plt.plot(pos, avint(pos))
		stdfilt = filtfilt(b, a, std)
		sint = UnivariateSpline(pos, stdfilt, s=None, k=4)
		plt.subplot(212)
		plt.plot(pos, std)
		plt.plot(pos, sint(pos))
		plt.savefig("gap/prof"+sname+".pdf")
		plt.clf()
		plsq = fitNoiseExpression(average, std, sname)
		
		plotFisherInfo(avint, sint, sname, combo)
		plotFisherInfoParam(avint, sint, sname, plsq)

	combo.save()