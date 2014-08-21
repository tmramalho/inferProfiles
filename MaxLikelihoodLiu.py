'''
Created on Oct 18, 2013

@author: tiago
'''

import numpy as np
import time
import random
from scipy.interpolate import UnivariateSpline
from LiuDatasetProcess import LiuDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MLLiu(object):
	'''
	classdocs
	'''


	def __init__(self, dataset, seed = 1234, xmin = 0, xmax = 1):
		'''
		Constructor
		'''
		self.dataset = dataset
		self.rng = np.random.RandomState(seed)
		random.seed(seed)
		self.xmin = xmin
		self.xmax = xmax
		
	
	def fitFlyLine(self, i, n = 10):
		x, y = self.dataset.returnPointCloudForFlyLine(i, True, self.xmin, self.xmax)
		self.data = (x, y)
		binAv, _ = fly.getBinnedDataForFlyLine(i, n, True, self.xmin, self.xmax)
		self.best = np.concatenate((binAv, [0.1, 0.1, 0.1]))
		start = time.clock()
		self.trainDifferentialEvolution(nGenerations = 10000)
		print "Done in", (time.clock() - start)*1000
	
	def evalTrainCost(self, x):
		if(x.min() <= 0):
			return 10000000
		points = x[:-3]
		se = x[-3]
		s = x[-2]
		p = x[-1]
		pts = self.data
		
		ap = np.linspace(self.xmin, self.xmax, len(points))
		m = UnivariateSpline(ap, points)
		
		penalty = 0
		derPoints = np.linspace(0, 1, 30)
		mDer = m(derPoints, 2)
		if(np.any(mDer < 0)):
			mPts = np.select([mDer < 0], [mDer])
			penalty = 5000*np.sqrt(np.dot(mPts,mPts))
		
		try:
			res = (np.power((pts[1]-m(pts[0])),2)/(2*s*(p*np.power(m(pts[0]),2)+m(pts[0])+se)) + 
					0.5*np.log(2*np.pi*s*(p*np.power(m(pts[0]),2)+m(pts[0])+se)))
		except FloatingPointError:
			return 1000000

		val = np.sum(res)
		if np.isnan(val):
			print x
			return 1000000
			
		return penalty + val
	
	def trainDifferentialEvolution(self, nGenerations = 1000, absVal = 1e-5, popSize = 20, F = 0.8, CR = 0.9):
		'''
		doesnt use the regularized cost
		'''
		settler = self.best
		ns = settler.shape[0]
		pop = settler + self.rng.normal(scale = 0.1, size=ns*popSize).reshape((popSize, ns))
		population = []
		scores = []
		for n in xrange(pop.shape[0]):
			cost = self.evalTrainCost(pop[n])
			population.append(pop[n])
			scores.append(cost)
		
		for _ in xrange(nGenerations):
			start = time.clock()
			for (j, p) in enumerate(population):
				targetIndex = self.rng.randint(0, ns)
				others = population[:j] + population[(j + 1):]
				(pa, pb, pc) = random.sample(others, 3)
				trial = np.copy(p)
				for n in xrange(ns):
					r = self.rng.rand()
					if n == targetIndex or r < CR:
						trial[n] = pa[n] + F*(pb[n] - pc[n])
				cost = self.evalTrainCost(trial)
				if cost < scores[j]:
					scores[j] = cost
					population[j] = trial
			print "New generation with score", min(scores), "up to", max(scores), "in", (time.clock() - start)*1000
		
		bestIndex = scores.index(min(scores))
		self.best = population[bestIndex]
		
		
		
if __name__ == '__main__':
	fly = LiuDataset()
	xmi = 0.1
	xma = 0.9
	ml = MLLiu(fly, xmin=xmi, xmax=xma)
	dosage = []
	params = []
	for line in range(fly.numLines):
		if line != 8:
			continue
		ml.fitFlyLine(line+1, 20)
		print "fit line", line+1
		points = ml.best[:-3]
		dosage.append(fly.returnMaxValueForFlyLine(line+1))
		se = ml.best[-3]
		s = ml.best[-2]
		p = ml.best[-1]
		params.append([s,p,se])
		ap = np.linspace(xmi, xma, len(points))
		m = UnivariateSpline(ap, points)
		sigma = lambda x: np.sqrt(s*(p*np.power(m(x),2)+m(x)+se))
		fi = lambda m, mp: 2*np.power(s * (p * np.power(m,2) + m),2)/(np.power(mp,2) * 
			(2 * s * (p * np.power(m,2) + m) + np.power(s * (2 * p * m + 1), 2)))
		fip = lambda m, mp: s * (p * np.power(m,2) + m + se) / np.power(mp,2)
		plt.figure(figsize=(10,20))
		gs = gridspec.GridSpec(3, 2)
		plt.suptitle("s0 "+str(s)+" p "+str(p)+" w "+str(se))
		plt.subplot(gs[0,0])
		plt.scatter(ml.data[0], ml.data[1], s=1, facecolor='0.5', lw = 0)
		plt.plot(ap, m(ap))
		plt.plot(ap, m(ap)+sigma(ap), color = 'r')
		plt.plot(ap, m(ap)-sigma(ap), color = 'r')
		plt.subplot(gs[0,1])
		plt.plot(ap, m(ap, 1))
		plt.twinx()
		plt.plot(ap, sigma(ap), color='r')
		plt.subplot(gs[1,1])
		plt.plot(ap, fi(m(ap), m(ap, 1)))
		plt.plot(ap, fip(m(ap), m(ap, 1)))
		plt.ylim([0,0.01])
		plt.subplot(gs[1,0])
		binAv, binSig = fly.getBinnedDataForFlyLine(line+1, 60, True, xmi, xma)
		plt.scatter(binAv, binSig)
		plt.plot(m(ap), sigma(ap))
		plt.subplot(gs[2,0])
		bx = np.linspace(xmi, xma, len(binAv))
		plt.scatter(bx, binAv)
		plt.plot(ap, m(ap))
		plt.subplot(gs[2,1])
		plt.scatter(bx, binSig)
		plt.plot(ap, sigma(ap))
		plt.savefig("results/liuFit"+str(line+1)+".pdf")
		plt.clf()
	params = np.array(params)
	plt.subplot(311)
	plt.scatter(dosage, params[:, 0])
	plt.subplot(312)
	plt.scatter(dosage, params[:, 1])
	plt.subplot(313)
	plt.scatter(dosage, params[:, 2])
	plt.savefig("results/dosagedependence.pdf")