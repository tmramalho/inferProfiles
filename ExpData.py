'''
Created on Sep 20, 2012

@author: tiago
'''

import numpy as np
import csv
import math
import matplotlib.pyplot as plt

class ExpData(object):
	'''
	Container for the experimental data
	'''

	def __init__(self, filename, bs = 20):
		'''
		Constructor
		'''
		if filename == 'gregor.csv':
			greg = csv.reader(open("data/gregor.csv", "rb"), delimiter=",")
			data = []
			self.scale = 400
			self.background = 30
			self.max = 500
			self.minx = 0
			self.maxx =  350
			self.sm = 200
			
			for row in greg:
				data.append(np.array(row, dtype=float))
			self.points = [np.empty([0]), np.empty([0])]
			
			for i in range(0, len(data)):
				if (i % 2 == 0):
					mp = data[i].min()
					mk = (data[i]-mp).max()
					data[i] = (data[i]-mp)/mk
					self.points[0] = np.concatenate((self.points[0], data[i]))
				else:
					data[i] = data[i]-self.background
					self.points[1] = np.concatenate((self.points[1], data[i]))
		elif filename == 'stas.csv' or filename == 'bomyi.csv' or filename == 'nc13.csv' or filename == 'nc14.csv':
			stas = csv.reader(open("data/" + filename, "rb"), delimiter=",")
			self.points = [np.empty([0]), np.empty([0])]
			
			self.scale = 180
			self.background = 0
			self.max = 220
			self.minx = 0
			self.maxx =  150
			self.sm = 40
			
			for row in stas:
				data = np.array(row, dtype=float)
				#dvSet = data[:50]
				vdSet = data[50:]
				dvAxis = np.linspace(0, 1, 50)
				#self.points[0] = np.concatenate((self.points[0], dvAxis))
				self.points[0] = np.concatenate((self.points[0], dvAxis))
				#self.points[1] = np.concatenate((self.points[1], dvSet[::-1]-self.background))
				self.points[1] = np.concatenate((self.points[1], vdSet-self.background))
		else:
			raise Exception('not implemented yet')

		#override default scale
		self.scale = self.points[1].max()
				
		#binned data
		self.numBins = bs
		bins = [[] for i in range(0, bs)]
		
		for i in range(0, len(self.points[0])):
			b = math.floor(self.points[0][i]*bs)
			if b >= bs:
				b = bs - 1
			bins[int(b)].append(self.points[1][i])
			
		self.avs = []
		self.stds = []
		
		for abin in bins:
			data = np.array(abin)
			self.avs.append(np.average(data))
			self.stds.append(np.std(data))
		
		self.avsNorm = self.avs / self.scale
		self.stdsNorm = self.stds / self.scale
			
	def plotData(self):
		plt.clf()
		plt.scatter(self.points[0],self.points[1]+self.background, c='b', marker='o', s=5)
		plt.savefig("results/expData.pdf")
		