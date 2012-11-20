'''
Created on Sep 20, 2012

@author: tiago
'''
import numpy as np
from random import Random
from time import time
from inspyred import ec
from inspyred.ec import terminators
from inspyred.ec import evaluators
from scipy.interpolate import UnivariateSpline

class MaxLikelihood(object):
	'''
	Optimize the likelihood for the gaussian model
	'''


	def __init__(self):
		'''
		Constructor
		'''
		prng = Random()
		prng.seed(time())
		self.ea = ec.DEA(prng)
		self.ea.terminator = (terminators.average_fitness_termination, terminators.evaluation_termination)
		self.ea.observer = self.observer
		
	def run(self, dataContainer, model = 'intext', genomeSize = 20):
		if model == 'intext':
			genFun = self.generate
			evalFun = evaluators.evaluator(self.logp)
		else:
			genFun = self.generateUnbias
			evalFun = evaluators.evaluator(self.logpUnbias)
		
		final_pop = self.ea.evolve(
						generator=genFun,
						evaluator=evalFun,
						pop_size=20,
						bounder=ec.Bounder(0, None),
						maximize=False,
						crossover_rate=0.9,
						mutation_rate=0.2,
						data=dataContainer.points,
						scale=dataContainer.scale,
						dim = genomeSize,
						tolerance = 0.0001,
						max_evaluations = 4000000)

		ind = max(final_pop)
		self.best = ind
		print ind.fitness
		
		return ind.candidate
	
	def logp(self, xl, args):
		x = np.array(xl)
		if(x.min() <= 0):
			return 10000000
		cp = x[:-3]
		se = x[-3]
		s = x[-2]
		p = x[-1]
		d = len(cp)
		tp = np.linspace(0, 1, d)
		m = UnivariateSpline(tp, cp)
		pts = args.get('data')
		
		try:
			res = (np.power((pts[1]-m(pts[0])),2)/(2*s*(p*np.power(m(pts[0]),2)+m(pts[0])+se)) + 
					0.5*np.log(2*3.14159265*s*(p*np.power(m(pts[0]),2)+m(pts[0])+se)))
		except FloatingPointError:
			return 1000000

		val = np.sum(res)
		if np.isnan(val):
			print x
			return 1000000
		return val

	def generate(self, random, args):
		d=args.get('dim')
		sc = args.get('scale')
		c=sc*np.random.rand(d)
		for i in range(0, d):
			c[i] = c[i]*(d-i)/d
		return np.concatenate((c,np.array([0.01*sc,0.01*sc, 5.0/sc]))).tolist()
	
	def logpUnbias(self, xl, args):
		x = np.array(xl)
		if(x.min() < 0):
			return 10000000
		tnsc = 2/args.get('scale')
		penalty = 0
		l = len(x)/2
		xPos = (l-2)/2 #fPos = (l-2)/2 + 2
		avProfilePoints = x[:l]
		avx = np.append(np.append([0], np.sort(np.tanh(tnsc*avProfilePoints[:xPos]))),[1])
		av = avProfilePoints[xPos:]
		sigmaProfilePoints = x[l:]
		sigmax = np.append(np.append([0], np.sort(np.tanh(tnsc*sigmaProfilePoints[:xPos]))),[1])
		sigma = sigmaProfilePoints[xPos:]
		m = UnivariateSpline(avx, av)
		s = UnivariateSpline(sigmax, sigma)
		pts = args.get('data')
		
		derPoints = np.linspace(0, 1, 27)
		mDer = m(derPoints, 1)
		sDer = s(derPoints, 1)
		if(np.any(mDer > 0) or np.any(sDer > 0)):
			mPts = np.select([mDer > 0], [mDer])
			sPts = np.select([sDer >0], [sDer])
			penalty = (np.dot(mPts,mPts) + np.dot(sPts, sPts))
		
		try:
			res = (np.power((pts[1]-m(pts[0])),2)/(2*s(pts[0])) + 
					0.5*np.log(2*3.14159265*s(pts[0])))
		except FloatingPointError:
			return 10000000

		val = np.sum(res)
		if np.isnan(val):
			return 10000000
		
		return penalty + val
	
	def generateUnbias(self, random, args):
		l=args.get('dim')/2
		sc = args.get("scale")
		xPos = (l-2)/2
		fPos = (l-2)/2 + 2
		c=sc*np.random.rand(fPos)
		for i in range(0, fPos):
			c[i] = c[i]*(fPos-i)/fPos
		x = np.linspace(0, 1, xPos+2) #uniform distr
		xFinal = sc/2*np.arctanh(x[1:-1]) #remove 0 and 1
		return np.concatenate((xFinal,c,xFinal,2*c)).tolist()
	
	def observer(self, population, num_generations, num_evaluations, args):
		'''Because we are minimising the lowest value is returned by the max()
		of the population. This comes from the inspyred library...'''
		alphaMale = max(population)
		print "Gen", num_generations, "(", num_evaluations, "evals). Best:", alphaMale.fitness
		
	def save(self, csvName, model):
		with open('results/results.dat', 'a') as f:
			f.write(csvName + ', ' + model + ', ' + str(self.best.fitness))
			for xval in self.best.candidate:
				f.write(', ' + str(xval))
			f.write('\n')
	
	def checkIfExists(self, csvName, model, ignore=False):
		if ignore == True:
			return None
		else:
			result = None
			try:
				with open('results/results.dat', 'r') as f:
					for line in f.readlines():
						values = line.split(",")
						print values[0], values[1]
						if values[0] == csvName and values[1].strip() == model:
							print "Found", csvName, "for model", model, "with", values[2]
							print  np.array(values[3:], dtype=float)
							return np.array(values[3:], dtype=float)
			except IOError:
				return None
			return result