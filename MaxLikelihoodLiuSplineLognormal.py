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


	def __init__(self, dataset, seed = 1234, xmin = 0, xmax = 1, mpen = 1e-1, spen = 1e4):
		'''
		Constructor
		'''
		self.dataset = dataset
		self.rng = np.random.RandomState(seed)
		random.seed(seed)
		self.xmin = xmin
		self.xmax = xmax
		self.mpen = mpen
		self.spen = spen
		
	
	def fitFlyLine(self, i, n = 10):
		self.n = n
		x, y = self.dataset.returnPointCloudForFlyLine(i, True, self.xmin, self.xmax)
		self.data = (x, np.log(y))
		binAv, stdAv = fly.getBinnedDataForFlyLine(i, n, True, self.xmin, self.xmax)
		sig = np.log(1 + np.power(stdAv,2)/np.power(binAv,2))
		mu = np.log(binAv) - 0.5*sig
		self.best = np.concatenate((mu, sig))
		start = time.clock()
		self.trainDifferentialEvolution(nGenerations = 6000)
		print "Done in", (time.clock() - start)*1000
	
	def evalTrainCost(self, x):
		p1 = x[:self.n]
		p2 = x[self.n:]
		pts = self.data
		
		ap = np.linspace(self.xmin, self.xmax, len(p1))
		m = UnivariateSpline(ap, p1, s=0)
		s = UnivariateSpline(ap, p2, s=0)
		
		penalty = 0
		npts = 30
		derPoints = np.linspace(0, 1, npts)
		mDer = m(derPoints, 2)
		sDer = s(derPoints, 2)
		penalty += self.mpen * np.dot(mDer, mDer) / npts
		penalty += self.spen * np.dot(sDer, sDer) / npts
		
		try:
			res = (np.power((pts[1]-m(pts[0])),2)/(2*s(pts[0])) +
					0.5*np.log(2*np.pi*s(pts[0])))
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
		pop = np.tile(settler, popSize).reshape((popSize, ns))
		pop = pop * self.rng.normal(loc = 1, scale = 0.4, size=ns*popSize).reshape((popSize, ns))
		pop[0] = settler
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
		n = 10
		ml.fitFlyLine(line+1, n)
		print "fit line", line+1
		p1 = ml.best[:n]
		p2 = ml.best[n:]
		dosage.append(fly.returnMaxValueForFlyLine(line+1))
		ap = np.linspace(xmi, xma, len(p1))
		m = UnivariateSpline(ap, p1, s=0)
		sigma = UnivariateSpline(ap, p2, s=0)
		fi = lambda x: 2 * np.power(sigma(x),2) / (2 * sigma(x) * np.power(m(x, 1), 2) + np.power(sigma(x, 1), 2))
		fip = lambda x: sigma(x) / np.power(m(x, 1), 2)
		av = lambda x: np.exp(m(x)+sigma(x)/2)
		std = lambda x: np.sqrt((np.exp(sigma(x))-1)*np.exp(2*m(x)+sigma(x)))
		plt.figure(figsize=(10,24))
		gs = gridspec.GridSpec(4, 2)
		plt.suptitle("spline fit for line {0:d}".format(line+1))
		plt.subplot(gs[0,0])
		plt.scatter(ml.data[0], np.exp(ml.data[1]), s=1, facecolor='0.5', lw = 0)
		plt.plot(ap, av(ap))
		plt.plot(ap, av(ap)+std(ap), color = 'r')
		plt.plot(ap, av(ap)-std(ap), color = 'r')
		plt.subplot(gs[0,1])
		plt.plot(ap, m(ap, 1))
		plt.twinx()
		plt.plot(ap, sigma(ap), color='r')
		plt.subplot(gs[1,1])
		plt.plot(ap, fi(ap))
		plt.plot(ap, fip(ap))
		plt.ylim([0,0.01])
		plt.subplot(gs[1,0])
		binAv, binSig = fly.getBinnedDataForFlyLine(line+1, 60, True, xmi, xma)
		plt.scatter(binAv, binSig)
		plt.plot(av(ap), std(ap))
		plt.subplot(gs[2,0])
		bx = np.linspace(xmi, xma, len(binAv))
		plt.scatter(bx, binAv)
		plt.plot(ap, av(ap))
		plt.subplot(gs[2,1])
		plt.scatter(bx, binSig)
		plt.plot(ap, std(ap))
		plt.subplot(gs[3,0])
		plt.plot(ap, m(ap, 1))
		plt.twinx()
		plt.plot(ap, sigma(ap, 1), color = 'r')
		plt.subplot(gs[3,1])
		plt.plot(ap, np.power(m(ap, 2),2))
		plt.twinx()
		plt.plot(ap, np.power(sigma(ap, 2),2), color = 'r')
		plt.savefig("results/liuFit"+str(line+1)+"_sp.pdf")
		plt.clf()
	'''
	params = np.array(params)
	plt.subplot(311)
	plt.scatter(dosage, params[:, 0])
	plt.subplot(312)
	plt.scatter(dosage, params[:, 1])
	plt.subplot(313)
	plt.scatter(dosage, params[:, 2])
	plt.savefig("results/dosagedependence.pdf")
	'''