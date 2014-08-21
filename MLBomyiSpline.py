'''
Created on Oct 18, 2013

@author: tiago
'''

import numpy as np
import time
import random
from scipy.interpolate import UnivariateSpline
import scipy.optimize as opt
from ExpData import ExpData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MLBomyi(object):
	'''
	classdocs
	'''


	def __init__(self, dataset, seed = 1234, xmin = 0, xmax = 1, mpen = 1e-1, spen = 1e3):
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
		
	
	def getBinnedData(self, bs = 50, norm = False, xmin = 0, xmax = 1):
		x = self.data[0]
		y = self.data[1]
		bins = [[] for i in range(0, bs)]
		
		for i in range(0, len(x)):
			b = np.floor((xmin-x[i])/(xmin-xmax)*bs)
			if b >= bs:
				b = bs - 1
			bins[int(b)].append(y[i])
			
		avs = []
		stds = []
		
		for abin in bins:
			data = np.array(abin)
			avs.append(np.average(data))
			stds.append(np.std(data))
		
		return np.array(avs), np.array(stds)
	
	def fitFlyLine(self, n = 10):
		self.n = n
		x, y = self.dataset.points
		self.data = (x, y/self.dataset.scale)
		binAv, stdAv = self.getBinnedData(n, True, self.xmin, self.xmax)
		self.best = np.concatenate((binAv, np.power(stdAv, 2)))
		start = time.clock()
		self.trainDifferentialEvolution(nGenerations = 2000)
		np.save("results/mlbestBom.npy", ml.best)
		#self.best = np.load("results/mlbestBom.npy")
		print "Done in", (time.clock() - start)*1000
	
	def evalTrainCost(self, x):
		if(x.min() <= 0):
			return 10000000
		p1 = x[:self.n]
		p2 = x[self.n:]
		pts = self.data
		
		ap = np.linspace(self.xmin, self.xmax, len(p1))
		m = UnivariateSpline(ap, p1, s=0)
		s = UnivariateSpline(ap, p2, s=0)
		
		penalty = 0
		npts = 30
		derPoints = np.linspace(0, 1, npts)
		''' use relative derivative '''
		ml = UnivariateSpline(ap, np.log(p1), s=0)
		sl = UnivariateSpline(ap, np.log(p2), s=0)
		mDer = ml(derPoints, 2)
		sDer = sl(derPoints, 2)
		penalty += self.mpen * np.dot(mDer, mDer) / npts
		penalty += self.spen * np.dot(sDer, sDer) / npts
		print penalty
		
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
	csvName = 'bomyi.csv'
	xmi = 0
	xma = 1
	dataContainer = ExpData(csvName, 50)
	ml = MLBomyi(dataContainer, mpen=1e-3, spen=1e-2)
	n = 11
	ml.fitFlyLine(n)
	p1 = ml.best[:n]
	p2 = ml.best[n:]
	ap = np.linspace(xmi, xma, len(p1))
	m = UnivariateSpline(ap, p1, s=0)
	sigma = UnivariateSpline(ap, p2, s=0)
	ap = np.linspace(xmi, xma, 100)
	fi = lambda x: 2 * np.power(sigma(x),2) / (2 * sigma(x) * np.power(m(x, 1), 2) + np.power(sigma(x, 1), 2))
	fip = lambda x: sigma(x) / np.power(m(x, 1), 2)
	res = opt.minimize(fi, 0.5)
	plt.figure(figsize=(10,24))
	plt.rcParams['axes.grid']=False
	plt.rcParams['axes.linewidth']=2
	plt.rcParams['xtick.major.size']=4
	plt.rcParams['ytick.major.size']=4
	plt.rcParams['xtick.major.width']=2
	plt.rcParams['ytick.major.width']=2
	plt.rcParams['axes.edgecolor']='555555'
	gs = gridspec.GridSpec(4, 2)
	plt.suptitle("spline fit")
	plt.subplot(gs[0,0])
	plt.scatter(ml.data[0], ml.data[1], s=2, facecolor='0.5', lw = 0)
	plt.plot(ap, m(ap))
	plt.plot(ap, m(ap)+np.sqrt(sigma(ap)), color = 'r')
	plt.plot(ap, m(ap)-np.sqrt(sigma(ap)), color = 'r')
	plt.xlim([0,1])
	plt.subplot(gs[0,1])
	plt.plot(ap, np.power(m(ap, 1), 2))
	plt.yscale('log')
	plt.twinx()
	plt.plot(ap, sigma(ap), color='r')
	plt.yscale('log')
	plt.subplot(gs[1,1])
	plt.plot(ap, fi(ap))
	plt.plot(ap, fip(ap))
	plt.ylim([0,0.01])
	plt.xlim([0,1])
	plt.subplot(gs[1,0])
	binAv, binSig = ml.getBinnedData(50, True, xmi, xma)
	plt.scatter(binAv, binSig)
	plt.plot(m(ap), np.sqrt(sigma(ap)))
	plt.subplot(gs[2,0])
	bx = np.linspace(xmi, xma, len(binAv))
	plt.scatter(bx, binAv)
	plt.plot(ap, m(ap))
	plt.subplot(gs[2,1])
	plt.scatter(bx, binSig)
	plt.plot(ap, np.sqrt(sigma(ap)))
	plt.xlim([0,1])
	plt.subplot(gs[3,0])
	plt.plot(ap, m(ap, 1))
	plt.twinx()
	plt.plot(ap, sigma(ap, 1), color = 'r')
	plt.xlim([0,1])
	plt.subplot(gs[3,1])
	plt.plot(ap, np.power(m(ap, 2),2))
	plt.twinx()
	plt.plot(ap, np.power(sigma(ap, 2),2), color = 'r')
	plt.xlim([0,1])
	plt.savefig("results/BomyiFit_sp.pdf")
	plt.clf()
	plt.figure()
	dd = ml.data
	plt.scatter(dd[0], dd[1], s=4, facecolor='0.5', lw = 0)
	plotLine = plt.plot(ap, m(ap))
	plt.plot(ap, m(ap)+np.sqrt(sigma(ap)), ls='--', c=plotLine[0].get_color())
	plt.plot(ap, m(ap)-np.sqrt(sigma(ap)), ls='--', c=plotLine[0].get_color())
	plt.ylim([0, 1.2])
	plt.twinx()
	color_cycle = plt.gca()._get_lines.color_cycle
	next(color_cycle)
	plt.plot(ap, np.sqrt(fi(ap)))
	plt.ylim([0, 0.06])
	plt.xlim([0,1])
	plt.savefig('results/BomyiFisher.pdf')
