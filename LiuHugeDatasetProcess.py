'''
Created on Oct 23, 2013

@author: tiago
'''

import scipy.io
import scipy.stats
from scipy.optimize import minimize, curve_fit
import scipy.interpolate as interpolate
import scipy.signal as signal
import time
import mpmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import sys

def scaling_fn(l, a, b, c):
	'''
	:param l: independent variable
	:param a: exponent
	:param b: amplitude
	:param c: mean
	:return: s = b*l^a + c
	'''

	return b*np.power(l, a) + c

class LiuHugeDataset(object):
	'''
	Load a mat file with data and convert it into a more pratical format
	'''


	def __init__(self, name = 'data/2XAAllEmbryos.mat', force = False, clip = 1):
		if force:
			self.createDataFile(name)
		else:
			try:
				self.points = np.load("data/liuDataHuge.npy", None)
			except IOError:
				self.createDataFile(name)
		
		self.frame = pd.DataFrame(self.points)
		self.frame.columns = ['embryoNum', 'cf', 'len', 'qua', 'sess', 'ap', 'lr', 'side', 'x', 'y']
		self.frame = self.frame[self.frame.y > clip]
				
	def createDataFile(self, name):
		'''
		Load the mat file and convert it
		A list of structs is converted into the following format:
			data['StructName'][0,N] -- with N the number of items
		to access the field names at each element, we can look at the dtypes
			data['StructName'][0,i].dtype
		to access a field value, just access its name
			data['StructName'][0,i]['FieldName']
		If the field is itself a list of structs,
		just apply this procedure recursively
		To create the data table I just create a list with all the points
		Not very memory efficient, but the original data is just 1Mb so any
		modern computer should handle it
		'''
		greg = scipy.io.loadmat(name, struct_as_record=True)
		embryos = greg['AllEmbryos_2XA']
		numEmbryos = embryos.shape[1]
		points = []
		for e in xrange(numEmbryos):
			cf = embryos['CF'][0][e][0][0]
			el = embryos['EggLength'][0][e][0][0]
			qu = embryos['Quality'][0][e][0][0]
			se = embryos['Session'][0][e][0][0]
			ap = embryos['Orientation'][0][e]['AP'][0][0][0][0]
			lr = embryos['Orientation'][0][e]['LR'][0][0][0][0]
			gl = embryos['Gradient'][0][e]['left'][0][0]
			gr = embryos['Gradient'][0][e]['right'][0][0]
			for p in gl:
				points.append([e, cf, el, qu, se, ap, lr, 0, p[0], p[1]])
			for p in gr:
				points.append([e, cf, el, qu, se, ap, lr, 1, p[0], p[1]])
		self.points = np.array(points)
		np.save("data/liuDataHuge.npy", self.points)
		
	def plotScatterQuality(self, quality = 1):
		x,y = self.returnPointCloudForQuality(quality)
		plt.scatter(x,y,s=1,facecolor='0.5', lw = 0)
		plt.show()
		
	def returnPointCloudForQuality(self, quality = 1, length = False):
		filtFrame = self.frame[(self.frame['qua'] <= quality)]
		if length:
			x = filtFrame['x']*filtFrame['len']
		else:
			x = filtFrame['x']
		y = filtFrame['y']
		return x, y
	
	def returnClippedPointCloudForQuality(self, quality = 1, length = False, minx=0):
		filtFrame = self.frame[(self.frame['qua'] <= quality) & (self.frame['x'] > minx)]
		if length:
			x = filtFrame['x']*filtFrame['len']
		else:
			x = filtFrame['x']
		y = filtFrame['y']
		return x, y
	
	def returnSidedPointCloud(self, side = 0, quality = 1, length = False):
		filtFrame = self.frame[(self.frame['qua'] <= quality) & (self.frame['side'] == side)]
		if length:
			x = filtFrame['x']*filtFrame['len']
		else:
			x = filtFrame['x']
		y = filtFrame['y']
		return x, y

	def get_length_percentile_binned_data(self, length_percentile, lower=True, num_bins=50, quality=1):
		filtFrame = self.frame[(self.frame['qua'] <= quality)]
		lengths = filtFrame.len
		if lower:
			perc = np.percentile(lengths, length_percentile)
			x = filtFrame[filtFrame.len <= perc]['x'].values
			y = filtFrame[filtFrame.len <= perc]['y'].values
		else:
			perc = np.percentile(lengths, 100-length_percentile)
			x = filtFrame[filtFrame.len >= perc]['x'].values
			y = filtFrame[filtFrame.len >= perc]['y'].values

		bins = [[] for i in range(0, num_bins)]

		for i in range(0, len(x)):
			b = np.floor(x[i]*num_bins)
			if b >= num_bins:
				b = num_bins - 1
			bins[int(b)].append(y[i])

		avs = []
		stds = []

		for abin in bins:
			npbin = np.array(abin)
			avs.append(np.average(npbin))
			stds.append(np.std(npbin))

		return bins, avs, stds


	def getIndividualProfiles(self, nump = 50, quality = 1, length = False, minx = 0):
		nEmbryos = int(self.frame.embryoNum.max() + 1)
		profiles = []
		lens = []
		counter = 0
		for n in xrange(nEmbryos):
			leftSide  = self.frame[(self.frame.embryoNum == n) &
								(self.frame.side == 0) &
								(self.frame.qua <= quality) &
								(self.frame.x > minx)]
			if leftSide.x.values.shape[0] > 0:
				if length:
					profiles.append([leftSide.x.values*leftSide.len/500, leftSide.y.values])
				else:
					profiles.append([leftSide.x.values, leftSide.y.values])
				lens.append([leftSide.len.values[0], leftSide.cf.values[0], n])
				counter += 1
			rightSide = self.frame[(self.frame.embryoNum == n) &
								(self.frame.side == 1) &
								(self.frame.qua <= quality) &
								(self.frame.x > minx)]
			if rightSide.x.values.shape[0] > 0:
				if length:
					profiles.append([rightSide.x.values*rightSide.len/500, rightSide.y.values])
				else:
					profiles.append([rightSide.x.values, rightSide.y.values])
				lens.append([rightSide.len.values[0], rightSide.cf.values[0], n])
				counter += 1
			if counter > nump-1:
				break
		return profiles, np.array(lens)
				
	def getBinnedData(self, quality = 1, bs = 50, length=False, normed=False, side = None):
		if side is not None:
			x, y = self.returnSidedPointCloud(side, quality, length)
		else:
			x, y = self.returnPointCloudForQuality(quality, length)
		x = x.values
		y = y.values
		bins = [[] for i in range(0, bs)]
		if length:
			mx = np.max(x)
		else:
			mx = 1
		
		for i in range(0, len(x)):
			b = np.floor(x[i]/mx*bs)
			if b >= bs:
				b = bs - 1
			bins[int(b)].append(y[i])
			
		avs = []
		stds = []
		
		for abin in bins:
			npbin = np.array(abin)
			if normed and npbin.shape[0] > 0:
				norm = np.max(npbin)
				avs.append(np.average(npbin/norm))
				stds.append(np.std(npbin/norm))
			else:
				avs.append(np.average(npbin))
				stds.append(np.std(npbin))
		
		return bins, avs, stds
		
	def plotHistogramsForBins(self, bs = 50, sbs = 30, quality = 1):
		bins, avs, stds = self.getBinnedData(quality, bs)
		nPlots = bs / 10
		gs = gridspec.GridSpec(1 + nPlots, 2)
		plt.subplot(gs[0, :])
		ap = np.linspace(0, 1, len(avs))
		plt.errorbar(ap, avs, stds)
		for i in range(nPlots * 2):
			plt.subplot(gs[i/2+1, i % 2])
			d = (i / 2) * 10
			m = i % 2
			bn = d + 1 + m * 7
			if len(bins[int(bn)]) == 0:
				continue
			_, bx, _ = plt.hist(bins[int(bn)], bins=sbs, label=str(bn)+":"+str(bn/float(bs)), normed = True)
			y = mlab.normpdf(bx, avs[bn], stds[bn])
			plt.plot(bx, y, 'k--', linewidth=1.5)
			plt.legend()
		plt.show()
		
	def plotChiForBins(self, bs = 50, sbs = 30, quality = 1):
		bins, avs, stds = self.getBinnedData(quality, bs)
		rv = scipy.stats.chi2(1)
		for bn in range(bs):
			if len(bins[bn]) == 0:
				continue
			chiY = np.power(bins[bn] - avs[bn], 2)/np.power(stds[bn],2)
			n, bx, _ = plt.hist(chiY, bins=sbs, label=str(bn)+":"+str(bn/float(bs)), normed = True)
			x = np.linspace(0, max(bx), 100)
			plt.plot(x, rv.pdf(x), 'g--', linewidth=2.5)
			ym = max(n)*1.1
			plt.ylim([0,ym])
			plt.legend()
			plt.savefig("results/chi2"+str(bn)+".pdf")
			plt.clf()
			
	def plotLogHistogramsForBins(self, bs = 50, sbs = 30, quality = 1, logBool = False):
		bins, avs, stds = self.getBinnedData(quality, bs)
		for bn in range(bs):
			if len(bins[bn]) == 0:
				continue
			if logBool:
				plt.yscale('log')
			_, bx, _ = plt.hist(bins[bn], bins=sbs, label=str(bn)+":"+str(bn/float(bs)), normed = True, log=logBool)
			y = mlab.normpdf(bx, avs[bn], stds[bn])
			plt.plot(bx, y, 'k--', linewidth=1.5)
			plt.legend()
			plt.savefig("results/normal"+str(bn)+".pdf")
			plt.clf()
			
	def plotHistogramsLogNormalForBins(self, bs = 50, sbs = 30, quality = 1):
		bins, avs, stds = self.getBinnedData(quality, bs)
		rv = scipy.stats.lognorm
		for bn in range(bs):
			if len(bins[bn]) == 0:
				continue
			_, bx, _ = plt.hist(bins[bn], bins=sbs, label=str(bn)+":"+str(bn/float(bs)), normed = True)
			mu = np.log(np.power(avs[bn],2)/np.sqrt(np.power(avs[bn],2) + np.power(stds[bn],2)))
			sig = np.log(np.sqrt(np.power(avs[bn],2) + np.power(stds[bn],2))/avs[bn])
			data = np.clip(bins[bn], 0.01, 1e10)
			shape, _, scale = rv.fit(data, floc=0)
			#print np.abs(shape-np.sqrt(sig))/shape, np.abs(scale - np.exp(mu))/scale
			plt.plot(bx, rv.pdf(bx, shape, scale=scale), 'k--', linewidth=1.5)
			plt.plot(bx, mlab.normpdf(bx, avs[bn], stds[bn]), 'r:', linewidth=1.5)
			plt.legend()
			plt.savefig("results/lognormal"+str(bn)+".pdf")
			plt.clf()
			
	def plotCorrectedHistogramsLogNormalForBins(self, bs = 50, sbs = 30, quality = 1):
		bins, avs, stds = self.getBinnedData(quality, bs, length=True)
		rv = scipy.stats.lognorm
		nbin = lambda k, p, r: mpmath.gamma(k + r)/(mpmath.gamma(k+1)*mpmath.gamma(r))*np.power(1-p, r)*np.power(p, k)
		nbin_npy = np.frompyfunc(nbin, 3, 1)
		for bn in range(bs):
			if len(bins[bn]) == 0:
				continue
			_, bx, _ = plt.hist(bins[bn], bins=sbs, label=str(bn)+":"+str(bn/float(bs)), normed = True)
			#mu = np.log(np.power(avs[bn],2)/np.sqrt(np.power(avs[bn],2) + np.power(stds[bn],2)))
			#sig = np.log(np.sqrt(np.power(avs[bn],2) + np.power(stds[bn],2))/avs[bn])
			data = np.clip(bins[bn], 0.01, 1e10)
			x = np.linspace(np.min(bx), np.max(bx), 100)
			shape, _, scale = rv.fit(data, floc=0)
			plt.plot(x, rv.pdf(x, shape, scale=scale), 'k--', linewidth=1.5, label = 'lognormal')
			plt.plot(x, mlab.normpdf(x, avs[bn], stds[bn]), 'r:', linewidth=1.5, label = 'normal')
			if(bn % 10 == 0):
				r = (avs[bn]*avs[bn])/(stds[bn]*stds[bn]-avs[bn])
				p = (stds[bn]*stds[bn]-avs[bn])/(stds[bn]*stds[bn])
				ssq = lambda params: -np.sum(np.log(nbin_npy(np.array(bins[bn]), params[0], params[1]).astype('float64')))
				res = minimize(ssq, np.array([p,r]), method='nelder-mead')
				print p,r,res
				plt.plot(x, nbin_npy(x, res.x[0], res.x[1]).astype('float64'), "g-", linewidth = 2, label='nbin')
			plt.legend()
			plt.savefig("results/clognormal"+str(bn)+".pdf")
			plt.clf()
			
	def plotAdjustedHistograms(self, bs = 50, sbs = 30, quality = 1):
		bins, avs, stds = self.getBinnedData(quality, bs, length=True)
		rv = scipy.stats.lognorm
		nbin = lambda k, p, r: mpmath.gamma(k + r)/(mpmath.gamma(k+1)*mpmath.gamma(r))*np.power(1-p, r)*np.power(p, k)
		nbin_npy = np.frompyfunc(nbin, 3, 1)
		results = []
		for bn in range(bs):
			if len(bins[bn]) == 0:
				continue
			start = time.clock()
			data = np.clip(bins[bn], 0.01, 1e10)*2000/128.12
			av = np.average(data)
			va = np.var(data)
			_, bx, _ = plt.hist(data, bins=sbs, label=str(bn)+":"+str(bn/float(bs)), normed = True)
			x = np.linspace(np.min(bx), np.max(bx), 100)
			shape, _, scale = rv.fit(data, floc=0)
			plt.plot(x, rv.pdf(x, shape, scale=scale), 'k--', linewidth=1.5, label = 'lognormal')
			plt.plot(x, mlab.normpdf(x, av, np.sqrt(va)), 'r:', linewidth=1.5, label = 'normal')
			r = (av*av)/(va-av)
			p = (va-av)/(va)
			ssq = lambda params: -np.sum(np.log(nbin_npy(data, params[0], params[1]).astype('float64')))
			res = minimize(ssq, np.array([p,r]), method='nelder-mead')
			#print p,r,res
			plt.plot(x, nbin_npy(x, res.x[0], res.x[1]).astype('float64'), "g-", linewidth = 2, label='nbin')
			plt.title(str(res.x[0]) + ", " + str(res.x[1]))
			results.append([bn, res.x[0], res.x[1]])
			plt.legend()
			plt.savefig("results/clognormal"+str(bn)+".pdf")
			plt.clf()
			print bn, "done in", time.clock() - start
		results = np.array(results)
		np.save("results/nbinParams.npy", results)
		
	def plotIndividualProfiles(self, nump = 50, quality = 1):
		profiles, _ = self.getIndividualProfiles(nump, quality)
		colors = cm.rainbow(np.linspace(0, 1, nump))
		for i,p in enumerate(profiles[:nump]):
			profile = interpolate.UnivariateSpline(p[0], p[1], s=1000000)
			x = np.linspace(0,0.9,100)
			plt.plot(x, profile(x), color=colors[i], alpha = 0.2)
		plt.savefig('results/profiles.png')
		
	def plotSidedProfiles(self, quality = 1, bs = 100):
		bins, av, st = self.getBinnedData(quality, bs, length=True, side=0)
		x = np.linspace(0, 1, len(av))
		plt.errorbar(x, np.array(av), yerr=np.array(st), fmt='r-')
		bins, av, st = self.getBinnedData(quality, bs, length=True, side=1)
		x = np.linspace(0, 1, len(av))
		plt.errorbar(x, np.array(av), yerr=np.array(st), fmt='b-')
		plt.savefig('results/sidecomp.pdf')
	
	def lookAtCrossCorrelation(self, quality = 1, bs = 100, nump = 50):
		bins, av, st = self.getBinnedData(quality, bs, length=True, side=0)
		s = np.linspace(0, 1, len(av))
		mu = interpolate.UnivariateSpline(s, np.nan_to_num(av), s=1000)
		sig = interpolate.UnivariateSpline(s, np.nan_to_num(st), s=100)
		profiles, _ = self.getIndividualProfiles(nump, quality)
		colors = cm.rainbow(np.linspace(0, 1, nump))
		for i,p in enumerate(profiles):
			x = p[0]
			meas = p[1]
			fluct = (mu(x) - meas)/sig(x)
			s = signal.correlate(fluct, fluct)
			w = np.linspace(-1,1,len(s))
			plt.plot(w, s, color=colors[i], alpha = 0.03)
		plt.savefig('results/cross.png')
		
	def fitExponentials(self, quality = 1, nump = 50, length=False):
		profiles, lens = self.getIndividualProfiles(nump, quality, length=length, minx=0.1)
		colors = cm.rainbow(np.linspace(0, 1, nump))
		print "this is gonna take a while..."
		la = []
		mainfig = plt.figure()
		ax = mainfig.add_subplot(111)
		for i,p in enumerate(profiles):
			subfig = plt.figure()
			subax = subfig.add_subplot(111)
			lpe = -4
			ssq = lambda params: np.sum(np.power(params[0]*np.exp(params[1]*p[0]) - p[1],2))
			ssqj = lambda params: np.array([np.sum(2*(params[0]*np.exp(params[1]*p[0]) -
											p[1])*np.exp(params[1]*p[0])),
								np.sum(2*(params[0]*np.exp(params[1]*p[0]) -
										p[1])*params[0]*p[0]*np.exp(params[1]*p[0]))])
			res = minimize(ssq, np.array([np.max(p[1])*1.5, lpe]), method='BFGS', jac=ssqj)
			w = np.linspace(0,1,100)
			ax.plot(w, res.x[0]*np.exp(res.x[1]*w), color=colors[i], alpha = 0.03)
			print i, res.x, res.fun, res.message
			la.append([res.x[0], res.x[1], lens[i, 0], lens[i, 1], lens[i, 2]])
			subax.plot(w, res.x[0]*np.exp(res.x[1]*w), color=colors[i])
			subax.scatter(p[0], p[1], color=colors[i])
			subfig.savefig("results/fit{0:03d}.pdf".format(i))
			plt.close(subfig)
		la = np.array(la)
		np.save("results/fitDist.npy", la)
		mainfig.savefig('results/exp.png')
		plt.clf()
		gs = gridspec.GridSpec(6, 6)
		plt.subplot(gs[-2:,2:])
		plt.hist(la[:, 1], 20)
		plt.subplot(gs[:-2, :2])
		hist, bin_edges = np.histogram(la[:, 0], bins=20)
		bw = bin_edges[1]-bin_edges[0]
		plt.barh(bin_edges[:-1], hist, height=bw)
		plt.xticks(rotation=30)
		plt.subplot(gs[:-2,2:])
		hist, xedges, yedges = np.histogram2d(la[:, 0], la[:, 1], bins=20, normed=True)
		plt.imshow(hist.T, interpolation = 'nearest', cmap=cm.Blues, aspect = 'auto',
				extent=[np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)],
				origin='lower')
		plt.grid(False)
		plt.savefig('results/lambda.png')
	
	def fitLogExponentials(self, quality = 1, nump = 50, length=False):
		profiles, lens = self.getIndividualProfiles(nump, quality, length=length, minx=0.1)
		colors = cm.rainbow(np.linspace(0, 1, nump))
		try:
			la = np.load("results/fitDistLog.npy")
		except IOError:
			print "this is gonna take a while..."
			la = []
			mainfig = plt.figure()
			ax = mainfig.add_subplot(111)
			for i,p in enumerate(profiles):
				#subfig = plt.figure()
				#subax = subfig.add_subplot(111)
				lpe = -4
				dx = p[0]
				dy = np.log(p[1])
				ssq = lambda params: np.sum(np.power(params[0] + params[1]*dx - dy,2))
				ssqj = lambda params: np.array(
						[np.sum(2*(params[0] + params[1]*dx - dy)),
						np.sum(2*(params[0] + params[1]*dx - dy)*dx)])
				res = minimize(ssq, np.array([np.log(np.max(p[1])*1.5), lpe]), method='BFGS', jac=ssqj)
				w = np.linspace(0,1,100)
				ax.plot(w, np.exp(res.x[0]+res.x[1]*w), color=colors[i], alpha = 0.03)
				print i, res.x, res.fun, res.message
				la.append([np.exp(res.x[0]), res.x[1], lens[i, 0], lens[i, 1], lens[i, 2]])
				#subax.plot(w, res.x[0]*np.exp(res.x[1]*w), color=colors[i])
				#subax.scatter(p[0], p[1], color=colors[i])
				#subfig.savefig("results/fit{0:03d}.pdf".format(i))
				#plt.close(subfig)
			la = np.array(la)
			np.save("results/fitDistLog.npy", la)
			mainfig.savefig('results/expLog.png')
			plt.clf()
		gs = gridspec.GridSpec(6, 6)
		plt.subplot(gs[:-2,2:])
		hist, xedges, yedges = np.histogram2d(la[:, 0], la[:, 1], bins=20)
		im = plt.imshow(hist, interpolation = 'nearest', cmap=cm.Blues, aspect = 'auto',
				extent=[np.min(yedges), np.max(yedges), np.min(xedges), np.max(xedges)],
				origin='lower')
		plt.grid(False)
		plt.subplot(gs[-2:,2:])
		plt.hist(la[:, 1], yedges)
		plt.xlim([np.min(yedges), np.max(yedges)])
		plt.grid(False)
		plt.subplot(gs[:-2, :2])
		hist, bin_edges = np.histogram(la[:, 0], bins=xedges)
		bw = bin_edges[1]-bin_edges[0]
		plt.barh(bin_edges[:-1], hist, height=bw)
		plt.ylim([np.min(xedges), np.max(xedges)])
		plt.xticks(rotation=30)
		plt.grid(False)
		ax = plt.subplot(gs[-1,0])
		plt.colorbar(im, cax=ax)
		plt.savefig('results/lambdaLog.pdf')
		plt.clf()
	
	def sampleLambdas(self, n_samples = 1000):
		la = np.load("results/fitDist.npy")
		x = np.linspace(0, 1, 100)
		hist, bin_edges = np.histogram(la[:, 1], bins=20, density=True)
		cum_values = np.zeros(bin_edges.shape)
		cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
		inv_cdf = interpolate.interp1d(cum_values, bin_edges)
		#plt.plot(x, inv_cdf(x))
		#plt.scatter(cum_values, bin_edges, color='r')
		#plt.show()
		r = np.random.rand(n_samples)
		samples = inv_cdf(r)
		c_av = np.average(la[:, 0])
		vals = c_av*np.exp(np.outer(x, samples))
		bins, _, _ = self.getBinnedData(2, 100)
		for i in xrange(100):
			if len(bins[i]) != 0:
				plt.hist(bins[i], bins=50, alpha=0.6)
			plt.hist(vals[i], bins=50, alpha=0.6, color='r')
			plt.savefig("results/comparehist{0:d}.pdf".format(i))
			plt.clf()
			
	def mvGaussian(self, pts, avs, cov):
		dim = avs.shape[0]
		invCov = np.linalg.inv(cov)
		av = np.tile(avs, (pts.shape[0], 1))
		'''The following allows us to calculate the dot products
		in the exp part of the gaussian for all pts simultaneously'''
		ip = np.dot(invCov, (pts - av).T).T # C^-1*(av-pt)
		dp = np.diag(np.inner(pts - av, ip)) # (av-pt)^T*C^-1*(av-pt)
		gauss = np.exp(-dp/2)*np.power(2*np.pi, -dim*0.5)*np.power(np.linalg.det(cov), -0.5)
		return gauss
	
	def sampleExponentials(self, n_samples = 1000):
		la = np.load("results/fitDistLog.npy")
		x = np.linspace(0, 1, 100)
		av = np.average(la[:, :2], axis=0)
		cov = np.cov(la[:, :2].T)
		s = np.random.multivariate_normal(av, cov, size=n_samples)
		vals = s[:, 0]*np.exp(np.outer(x, s[:, 1]))
		print vals.shape
		bins, _, _ = self.getBinnedData(2, 100)
		'''for i in xrange(100):
			if len(bins[i]) != 0:
				plt.hist(bins[i], bins=50, alpha=0.6, normed=True)
			plt.hist(vals[i], bins=50, alpha=0.6, color='r', normed=True)
			plt.savefig("results/comparefullhist{0:03d}.pdf".format(i))
			plt.clf()'''
		plt.subplot(231)
		hist, xedges, yedges = np.histogram2d(la[:, 0], la[:, 1], bins=20, normed=True)
		plt.imshow(hist.T, interpolation = 'nearest', cmap=cm.Blues, aspect = 'auto',
				extent=[np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)],
				origin='lower')
		plt.subplot(232)
		samples = np.random.multivariate_normal(av, cov, size=10*n_samples)
		hist_sp, xedges_sp, yedges_sp = np.histogram2d(samples[ :, 0], 
									samples[ :, 1], bins=50, normed=True)
		plt.imshow(hist_sp.T, interpolation = 'bicubic', cmap=cm.Reds, aspect = 'auto',
				extent=[np.min(xedges_sp), np.max(xedges_sp), np.min(yedges_sp), np.max(yedges_sp)],
				origin='lower')
		plt.subplot(233)
		plt.pcolor(np.linspace(np.min(xedges), np.max(xedges), len(xedges)),
				np.linspace(np.min(yedges), np.max(yedges), len(yedges)), hist.T,
				alpha=0.8, cmap=cm.Blues)
		plt.pcolor(np.linspace(np.min(xedges_sp), np.max(xedges_sp), len(xedges_sp)),
				np.linspace(np.min(yedges_sp), np.max(yedges_sp), len(yedges_sp)), hist_sp.T,
				alpha=0.5, cmap=cm.Reds)
		plt.subplot(234)
		plt.hist2d(la[:, 0], la[:, 1], bins=20, cmap=cm.Blues)
		plt.subplot(235)
		_, bx, _ = plt.hist(la[:, 1], 20, alpha=0.6, normed=True)
		plt.hist(samples[:, 1], 20, color='r', alpha=0.6, normed=True)
		xp = np.linspace(np.min(bx), np.max(bx), 100)
		p = scipy.stats.norm.pdf(xp, loc=av[1], scale=np.sqrt(cov[1,1]))
		plt.plot(xp, p)
		plt.subplot(236)
		nx, bx, _ = plt.hist(la[:, 0], 20, alpha=0.6, normed=True)
		plt.hist(samples[:, 0], 20, color='r', alpha=0.6, normed=True)
		xp = np.linspace(np.min(bx), np.max(bx), 100)
		p = scipy.stats.norm.pdf(xp, loc=av[0], scale=np.sqrt(cov[0,0]))
		plt.plot(xp, p)
		plt.twinx()
		plt.plot(bx[1:]-np.diff(bx), np.cumsum(nx)*np.diff(bx), color='g')
		plt.savefig("results/comparehistdist.pdf")
		plt.clf()

	def calculateError(self, n_samples):
		la = np.load("results/fitDistLog.npy")
		x = np.linspace(0, 520, 100)
		av = np.average(la[:, :2], axis=0)
		cov = np.cov(la[:, :2].T)
		s = np.random.multivariate_normal(av, cov, size=n_samples)
		L = np.random.normal(500, 0.001, size=n_samples)
		vals = s[:, 0]*np.exp(np.outer(x, s[:, 1]/L))
		plt.plot(np.std(vals, axis=1))
		s = np.random.multivariate_normal(av, np.diag(np.diag(cov)), size=n_samples)
		vals = s[:, 0]*np.exp(np.outer(x, s[:, 1]/L))
		print np.average(vals, axis=1)
		plt.plot(np.std(vals, axis=1))
		plt.savefig('results/error.png')

	def correlateExponentialsLength(self):
		la = np.load("results/fitDistLog.npy")
		plt.figure(figsize=(6*2, 4*2))
		x = np.linspace(np.min(la[:, 2]), np.max(la[:, 2]), 200)
		plt.subplot(221)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(la[:, 2], la[:, 0])
		print slope, intercept, r_value, p_value, std_err
		plt.scatter(la[:, 2], la[:, 0])
		plt.plot(x, slope*x+intercept, color='r')
		plt.title("A R^2:{0:.2f}, pval:{1:.2e}".format(r_value**2, p_value))
		plt.subplot(222)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(la[:, 2], la[:,0]/(la[:, 2]/500))
		print slope, intercept, r_value, p_value, std_err
		plt.scatter(la[:, 2], la[:,0]/(la[:, 2]/np.mean(la[:, 2])))
		plt.plot(x, slope*x+intercept, color='r')
		plt.title("A/L R^2:{0:.2e}, pval:{1:.2e}".format(r_value**2, p_value))
		plt.subplot(223)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(la[:, 2], la[:, 1])
		print slope, intercept, r_value, p_value, std_err
		plt.scatter(la[:, 2], la[:, 1])
		plt.plot(x, slope*x+intercept, color='r')
		plt.title("lambda R^2:{0:.2f}, pval:{1:.2e}".format(r_value**2, p_value))
		plt.subplot(224)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(la[:, 2], la[:,1]/(la[:, 2]/500))
		print slope, intercept, r_value, p_value, std_err
		plt.scatter(la[:, 2], la[:,1]/(la[:, 2]/np.mean(la[:, 2])))
		plt.plot(x, slope*x+intercept, color='r')
		plt.title("lambda/L R^2:{0:.2e}, pval:{1:.2e}".format(r_value**2, p_value))
		plt.savefig("results/scalingScatter.pdf")
		plt.clf()

	def fitScalingParameters(self):
		la = np.load("results/fitDistLog.npy")
		x = np.linspace(np.min(la[:, 2]), np.max(la[:, 2]), 200)
		plt.subplot(311)
		amp = np.log(la[:, 0])
		plt.scatter(la[:, 2], amp)
		popt, pcov = curve_fit(scaling_fn, la[:, 2], amp, p0 = [0, 1, np.mean(amp)])
		plt.plot(x, scaling_fn(x, *popt), color='r')
		print popt, np.mean(amp)
		plt.subplot(312)
		lam = la[:, 1]
		plt.scatter(la[:, 2], lam)
		popt, pcov = curve_fit(scaling_fn, la[:, 2], lam, p0 = [1, -0.01, -0.08], maxfev=20000)
		plt.plot(x, scaling_fn(x, *popt), color='r')
		print popt, np.mean(lam)
		plt.savefig("results/scalingFit.pdf")
		plt.clf()

	def pairWiseCorrelation(self):
		la = np.load("results/fitDistLog.npy")
		corr = np.zeros(6)
		av = np.average(la[:, :2], axis=0)
		for i,j in enumerate(la[:, 4]):
			try:
				if j == la[i+1, 4]:
					corr[0] += (la[i, 0] - av[0])*(la[i+1, 0] - av[0])
					corr[1] += (la[i, 1] - av[1])*(la[i+1, 1] - av[1])
					corr[2] += 1
				corr[3] += (la[i, 0] - av[0])*(la[i+1, 0] - av[0])
				corr[4] += (la[i, 1] - av[1])*(la[i+1, 1] - av[1])
				corr[5] += 1
			except IndexError:
				continue
		print corr[0]/corr[2], corr[3]/corr[5]
		print corr[1]/corr[2], corr[4]/corr[2]

if __name__ == '__main__':
	self = LiuHugeDataset()
	#self.plotHistogramsLogNormalForBins(quality=2, bs=100, sbs=50)
	#self.plotIndividualProfiles(nump = 200)
	#self.plotCorrectedHistogramsLogNormalForBins(100, 50, 2)
	#self.plotAdjustedHistograms(100, 50, 2)
	#self.plotSidedProfiles(quality=2)
	#self.lookAtCrossCorrelation(2, 100, 1000)
	#self.fitLogExponentials(1, 6000, False)
	#self.sampleExponentials(10000)
	self.calculateError(1000)
	#self.correlateExponentialsLength()
	#self.pairWiseCorrelation()
	#self.fitScalingParameters()