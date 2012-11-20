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


	def __init__(self, yp, sm=200):
		'''
		Constructor
		'''
		self.yp = yp
		self.xp = xp = np.linspace(0, 1, len(yp))
		self.m = UnivariateSpline(xp, yp, k=4, s=sm)

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
		plt.scatter(self.xp,self.yp)
		plt.plot(tp,self.m(tp))
		plt.savefig("results/splineFit.pdf")
		if order > 0:
			plt.plot(tp,self.m(tp,1))
			if order > 1:
				plt.plot(tp,self.m(tp,2))
			plt.savefig("results/splineDerivative.pdf")
	
	def plotSplineData(self, dataContainer, s, p, se, yscale):
		plt.clf()
		plt.xlim(0,1)
		plt.ylim(0,yscale)
		tp = np.linspace(0, 1, 100)
		plt.scatter(dataContainer.points[0],dataContainer.points[1]+dataContainer.background, c='b', marker='o', s=5)
		plt.plot(tp, self.m(tp)+dataContainer.background,'r', linewidth=2)
		plt.plot(tp, self.m(tp)+np.sqrt(s*(p*self.m(tp)*self.m(tp)+self.m(tp)+se))+dataContainer.background,'r--', linewidth=2)
		plt.plot(tp, self.m(tp)-np.sqrt(s*(p*self.m(tp)*self.m(tp)+self.m(tp)+se))+dataContainer.background,'r--', linewidth=2)
		plt.plot(tp, np.zeros(100) + dataContainer.background, '--', c='#BBBBBB', alpha=0.8)
		plt.savefig("results/splineVsData.pdf")
		
	def plotBinnedData(self, dataContainer, s, p, se, xmin, xmax):
		plt.clf()
		sigma = lambda x: s * (p*x*x + x + se)
		t = np.linspace(xmin, xmax, 100)
		plt.xlim(xmin, xmax)
		plt.plot(t, sigma(t))
		plt.plot(dataContainer.avs, np.power(dataContainer.stds,2), 'o')
		plt.savefig("results/noiseVsBins.pdf")
		plt.clf()
		tp = np.linspace(0, 1, dataContainer.numBins)
		plt.plot(tp, self.m(tp),'r', linewidth=2)
		plt.plot(tp, dataContainer.avs)
		plt.savefig("results/splineVsBins.pdf")
	
	def plotFisherInfo(self, dataContainer, s, p, se, ymax):
		plt.clf()
		t = np.linspace(0, 1, 100)
		
		minf = lambda x: -1 * self.m(x)
		minx = fminbound(minf, 0, 1)
		fval = self.m(minx)
		s = s/fval
		se = se/fval
		p = p*fval
		print fval, s, p
		fi = lambda j, jp: 1/(np.power(jp, 2) * (s + 2 * j * (1 + p * j) * (1 + 2 * p * s))
							/(2 * np.power(j, 2) * np.power(1 + p * j, 2) * s))
		fiapp = lambda j, jp: (4 * (j + p * j * j + se) * s)/(np.power(jp, 2))
		plt.xlim(0, 1)
		plt.ylim(0, ymax)
		plt.plot(t, fi(self.m(t)/fval, self.m(t, 1)/fval))
		plt.plot(t, fiapp(self.m(t)/fval, self.m(t, 1)/fval), 'r')
		plt.savefig("results/fisherInfo.pdf")