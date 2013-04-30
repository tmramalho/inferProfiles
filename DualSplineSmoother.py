'''
Created on Oct 2, 2012

@author: tiago
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

class DualSplineSmoother(object):
	'''
	claselfdocs
	'''


	def __init__(self, yp, workdir, scale, sm=200):
		'''
		Constructor
		'''
		yp = np.array(yp)
		self.l = len(yp)/2
		self.xPos = (self.l-2)/2 #fPos = (self.l-2)/2 + 2
		tnsc = 2/scale
		print tnsc
		plt.rcParams['font.size'] = 24
		plt.rcParams['lines.linewidth'] = 2.4
		self.workdir = workdir
		
		avProfilePoints = yp[:self.l]
		self.avx = np.append(np.append([0], np.sort(np.tanh(tnsc*avProfilePoints[:self.xPos]))),[1])
		self.av = avProfilePoints[self.xPos:]
		
		sigmaProfilePoints = yp[self.l:]
		self.sigmax = np.append(np.append([0], np.sort(np.tanh(tnsc*sigmaProfilePoints[:self.xPos]))),[1])
		self.sigma = sigmaProfilePoints[self.xPos:]
		
		self.m = UnivariateSpline(self.avx, self.av)
		print "Created spline with " + str(len(self.m.get_knots())) + " knots"

		self.s = UnivariateSpline(self.sigmax, self.sigma)
		print "Created spline with " + str(len(self.s.get_knots())) + " knots"

	def saveSpline(self, filename):
		tp = np.linspace(0, 1, 1000)
		with open(filename ,"w+") as f:
			for i in range(0, 1000):
				f.write( str(tp[i]) + " , " + str(self.m(tp[i])) )
				if i < 999:
					f.write("\n")
		f.close()
	
	def saveSigmaSpline(self, filename):
		tp = np.linspace(0, 1, 1000)
		with open(filename ,"w+") as f:
			for i in range(0, 1000):
				f.write( str(tp[i]) + " , " + str(self.s(tp[i])) )
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
		plt.subplot(211)
		plt.scatter(self.avx,self.av)
		plt.plot(tp,self.m(tp))
		plt.subplot(212)
		plt.scatter(self.sigmax,self.sigma)
		plt.plot(tp,self.s(tp))
		plt.savefig(self.workdir+"/splineFit.pdf")
		if order > 0:
			plt.clf()
			plt.subplot(211)
			plt.plot(tp,self.m(tp,1))
			plt.subplot(212)
			plt.plot(tp,self.s(tp,1))
			plt.savefig(self.workdir+"/splineDerivative.pdf")
	
	def plotSplineData(self, dataContainer, yscale):
		plt.clf()
		plt.xlim(0,1)
		plt.ylim(0,yscale)
		tp = np.linspace(0, 1, 100)
		plt.scatter(dataContainer.points[0],dataContainer.points[1]+dataContainer.background, c='b', marker='o', s=5)
		plt.plot(tp, self.m(tp)+dataContainer.background,'r', linewidth=2)
		plt.plot(tp, self.m(tp)+np.sqrt(self.s(tp))+dataContainer.background,'r--', linewidth=2)
		plt.plot(tp, self.m(tp)-np.sqrt(self.s(tp))+dataContainer.background,'r--', linewidth=2)
		plt.plot(tp, np.zeros(100) + dataContainer.background, '--', c='#BBBBBB', alpha=0.8)
		plt.savefig(self.workdir+"/splineVsData.pdf")
		
	def plotBinnedData(self, dataContainer):
		plt.clf()
		tp = np.linspace(0, 1, dataContainer.numBins)
		tpHD = np.linspace(0, 1, 500)
		plt.plot(self.m(tpHD), self.s(tpHD))
		plt.plot(dataContainer.avs, np.power(dataContainer.stds,2), 'o')
		plt.savefig(self.workdir+"/noiseVsBins.pdf")
		plt.clf()
		plt.plot(tpHD, self.m(tpHD),'r', linewidth=2)
		plt.plot(tp, dataContainer.avs, 'o')
		plt.savefig(self.workdir+"/splineVsBins.pdf")
		plt.clf()
		plt.plot(tpHD, self.s(tpHD),'r', linewidth=2)
		plt.plot(tp, np.power(dataContainer.stds,2), 'o')
		plt.savefig(self.workdir+"/spatialNoiseVsBins.pdf")
	
	def plotFisherInfo(self, dataContainer, ymax, ymaxsq):
		plt.clf()
		t = np.linspace(0, 1, 500)
		
		minf = lambda x: -1 * self.m(x)
		minx = fminbound(minf, 0, 1)
		fval = self.m(minx)
		
		noisemean = UnivariateSpline(self.m(t)/fval, self.s(t)/fval/fval)
		self.se = noisemean(0)
		
		fi = lambda a, sa, sp: 2*np.power(sa, 2)/ (np.power(a,2)*2*sa+np.power(sp,2))
		fiapp = lambda a, sa, sp: sa / (np.power(a,2))
		plt.xlim(0, 1)
		plt.ylim(0, ymaxsq)
		print 'whop whop'
		plt.plot(t, fi(self.m(t,1)/fval, self.s(t)/fval/fval-self.se, self.s(t, 1)/fval/fval))
		plt.plot(t, fiapp(self.m(t,1)/fval, self.s(t)/fval/fval-self.se, self.s(t, 1)/fval/fval), 'r')
		plt.savefig(self.workdir+"/variance.pdf")
		plt.clf()
		plt.ylim(0, ymax)
		plt.plot(t, np.sqrt(fi(self.m(t,1)/fval, self.s(t)/fval/fval-self.se, self.s(t, 1)/fval/fval)))
		plt.plot(t, np.sqrt(fiapp(self.m(t,1)/fval, self.s(t)/fval/fval-self.se, self.s(t, 1)/fval/fval)), 'r')
		plt.savefig(self.workdir+"/stddev.pdf")
		plt.clf()
		plt.plot(t, 1/fi(self.m(t,1)/fval, self.s(t)/fval/fval-self.se, self.s(t, 1)/fval/fval))
		plt.plot(t, 1/fiapp(self.m(t,1)/fval, self.s(t)/fval/fval-self.se, self.s(t, 1)/fval/fval), 'r')
		plt.savefig(self.workdir+"/fisherInfo.pdf")