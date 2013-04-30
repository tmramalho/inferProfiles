'''
Created on Feb 5, 2012

@author: tiago
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

class SplineSmoother(object):
	'''
	claselfdocs
	'''


	def __init__(self, yp, workdir, scale, sm=200):
		'''
		Constructor
		'''
		
		self.workdir = workdir
		yp = np.array(yp)
		self.l = len(yp)
		self.xPos = (self.l-2)/2 #fPos = (self.l-2)/2 + 2
		tnsc = 2/scale
		print tnsc
		plt.rcParams['font.size'] = 24
		plt.rcParams['lines.linewidth'] = 2.4
		self.workdir = workdir
		
		self.avx = np.append(np.append([0], np.sort(np.tanh(tnsc*yp[:self.xPos]))),[1])
		self.av = yp[self.xPos:]
		self.m = UnivariateSpline(self.avx, self.av)
		
		plt.rcParams['font.size'] = 24
		plt.rcParams['lines.linewidth'] = 2.4

		print "Created spline with " + str(len(self.m.get_knots())) + " knots"

	def saveSpline(self, filename):
		tp = np.linspace(0, 1, 1000)
		with open(filename ,"w+") as f:
			for i in range(0, 1000):
				f.write( str(tp[i]) + " , " + str(self.m(tp[i])) )
				if i < 999:
					f.write("\n")
		f.close()
	
	def showSpline(self, order=0):
		plt.clf()
		print "Spline full information:"
		print self.m.get_knots()
		print self.m.get_coeffs()
		print self.m.get_residual()
		tp = np.linspace(0, 1, 1000)
		plt.scatter(self.avx, self.av)
		plt.plot(tp,self.m(tp))
		plt.savefig(self.workdir+"/splineFit.pdf")
		if order > 0:
			plt.plot(tp,self.m(tp,1))
			if order > 1:
				plt.plot(tp,self.m(tp,2))
			plt.savefig(self.workdir+"/splineDerivative.pdf")
	
	def plotSplineData(self, dataContainer, s, p, se, yscale):
		plt.clf()
		plt.xlim(0,1)
		plt.ylim(0,yscale)
		tp = np.linspace(0, 1, 500)
		plt.scatter(dataContainer.points[0],dataContainer.points[1]+dataContainer.background, c='b', marker='o', s=5)
		plt.plot(tp, self.m(tp)+dataContainer.background,'r', linewidth=2)
		plt.plot(tp, self.m(tp)+np.sqrt(s*(p*self.m(tp)*self.m(tp)+self.m(tp)+se))+dataContainer.background,'r--', linewidth=2)
		plt.plot(tp, self.m(tp)-np.sqrt(s*(p*self.m(tp)*self.m(tp)+self.m(tp)+se))+dataContainer.background,'r--', linewidth=2)
		plt.plot(tp, np.zeros(500) + dataContainer.background, '--', c='#BBBBBB', alpha=0.8)
		plt.savefig(self.workdir+"/splineVsData.pdf")
		
	def plotBinnedData(self, dataContainer, s, p, se, xmin, xmax):
		plt.clf()
		sigma = lambda x: s * (p*x*x + x + se)
		t = np.linspace(xmin, xmax, 500)
		plt.xlim(xmin, xmax)
		plt.plot(t, sigma(t))
		plt.plot(dataContainer.avs, np.power(dataContainer.stds,2), 'o')
		plt.savefig(self.workdir+"/noiseVsBins.pdf")
		plt.clf()
		tp = np.linspace(0, 1, dataContainer.numBins)
		plt.plot(tp, self.m(tp),'r', linewidth=2)
		plt.plot(tp, dataContainer.avs, 'o')
		plt.savefig(self.workdir+"/splineVsBins.pdf")
	
	def plotFisherInfo(self, dataContainer, s, p, se, ymax, ymaxsq):
		plt.clf()
		t = np.linspace(0, 1, 1000)
		
		minf = lambda x: -1 * self.m(x)
		minx = fminbound(minf, 0, 1)
		fval = self.m(minx)
		s = s/fval
		se = se/fval
		p = p*fval
		print fval, s, p, se
		fi = lambda m, mp: 2*np.power(s * (p * np.power(m,2) + m),2)/(np.power(mp,2) * 
		(2 * s * (p * np.power(m,2) + m) + np.power(s * (2 * p * m + 1), 2)))
		fiapp = lambda m, mp: s * (p * np.power(m,2) + m) / np.power(mp,2)
		plt.xlim(0, 1)
		plt.ylim(0, ymaxsq)
		plt.plot(t, fi(self.m(t)/fval, self.m(t, 1)/fval))
		plt.plot(t, fiapp(self.m(t)/fval, self.m(t, 1)/fval), 'r')
		plt.savefig(self.workdir+"/variance.pdf")
		plt.clf()
		plt.ylim(0, ymax)
		plt.plot(t, np.sqrt(fi(self.m(t)/fval, self.m(t, 1)/fval)))
		plt.plot(t, np.sqrt(fiapp(self.m(t)/fval, self.m(t, 1)/fval)), 'r')
		plt.savefig(self.workdir+"/stddev.pdf")
		plt.clf()
		plt.plot(t, 1/fi(self.m(t)/fval, self.m(t, 1)/fval))
		plt.plot(t, 1/fiapp(self.m(t)/fval, self.m(t, 1)/fval), 'r')
		plt.savefig(self.workdir+"/fisherInfo.pdf")
		with open(self.workdir+"/variance.csv" ,"w+") as f:
			fisher=np.sqrt(fi(self.m(t)/fval, self.m(t, 1)/fval))
			for i in range(0, 1000):
				f.write( str(t[i]) + " , " + str(fisher[i]) )
				if i < 999:
					f.write("\n")
		f.close()