'''
Created on Oct 17, 2013

@author: tiago
'''

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class LiuDataset(object):
	'''
	Load a mat file with data and convert it into a more pratical format
	'''


	def __init__(self, name = 'data/Liu et al. - LiveImaging_data_structure.mat', force = False):
		if force:
			self.createDataFile(name)
		else:
			try:
				self.points = np.load("data/liuData.npy", None)
			except IOError:
				self.createDataFile(name)
		
		self.frame = pd.DataFrame(self.points)
		self.frame.columns = ['flyline', 'enum', 'cf', 'len', 'x', 'y']
		self.numLines = int(self.frame['flyline'].max())
				
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
		
		Readme from liu et al for easy reference: LiveImaging_data_structure.mat
		contains a data structure named "FlyLines" with Bicoid-GFP intensity gradient
		and cephalic furrow (CF) position measurements of fly embryos from live two-
		photon imaging for 29 fly lines.

		Each record in the structure contains the following fields: 1) FlyLineNumber:
		fly line number as listed in column 1 in Table S1 of Liu et al. PNAS 2013. 2)
		FlyLineName: fly line name as listed in column 2 of Table S1 of Liu et al.
		PNAS 2013. 3) Embryos: structure that contains the source data for one
		representative live imaging session. Size of this structure equals the number
		of Embryos N imaged in this session. ---Embryos.Gradient.left and
		Em.Gradient.right contain data for N gradients on left and right sides of the
		center line. Each has two columns: column1 contains the position of each
		detected nucleus in unit of egg length; column2 contains the apparent nuclear
		intensity of each nucleus after background subtraction. Bcd gradients were
		measured in the mid-coronal plane under dorsal view at about 16 min into
		nuclear cycle 14. ---Embryos.CF contains N cephalic furrow positions in unit
		of egg length measured ~50 min after Bcd gradient measurement.NaN means no
		data is available (typically because only the gradients was measured for this
		particular embryo). ---Embryos.EggLength: Egg lengths for N embryos in
		micrometer. ---Embryos.Orientation.LR: Rotational asymmetry around the left-
		right axis. Three classes of embryos identified based on the range of faint
		membrane segments: LR=0 (<1%EL), LR=1 or -1(<3%EL), and LR=2 or -2 (>3%EL).
		LR<0 and LR>0 indicates that the faint membrane segments are in the anterior
		pole and posterior pole, respectively. Only the embryos with LR=0 were
		selected for further data analysis. (See Liu et al. PNAS 2013.)
		---Embryos.Orientation.AP: Rotational asymmetry around the AP axis. Two groups
		of embryos identified based on shape symmetry. AP=1 if the image plane is
		closer to midsagittal plane, and  AP=0 if it is closer to the coronal plane.
		Only embryos with AP=0 were selected for further data analysis. (See Liu et
		al. PNAS 2013.)

		'''
		greg = scipy.io.loadmat(name, struct_as_record=True)
		FlyLines = greg['FlyLines']
		numLines = FlyLines.shape[1]
		pointList = []
		self.lineNameKeys = dict()
		for i in xrange(numLines):
			lineName = FlyLines[0,i]['FlyLineName']
			lineNumber = int(FlyLines[0,i]['FlyLineNumber'])
			self.lineNameKeys[lineNumber] = str(lineName)
			embryoList = FlyLines[0,i]['Embryos']
			numEmbryos = embryoList.shape[1]
			for j in xrange(numEmbryos):
				embCF = float(embryoList[0,j]['CF'])
				embLen = float(embryoList[0,j]['EggLength'])
				embRotAP = int(embryoList[0,j]['Orientation']['AP'])
				embRotLR = int(embryoList[0,j]['Orientation']['LR'])
				pointsLeft = embryoList[0,j]['Gradient']['left'][0][0]
				pointsRight = embryoList[0,j]['Gradient']['right'][0][0]
				for p in pointsLeft:
					if embRotLR == 0 and embRotAP == 0:
						pointList.append([lineNumber, j, embCF, embLen, p[0], p[1]])
				for p in pointsRight:
					if embRotLR == 0 and embRotAP == 0:
						pointList.append([lineNumber, j, embCF, embLen, p[0], p[1]])
		self.points = np.array(pointList)
		np.save("data/liuData.npy", self.points)
		
	def returnPointCloudForFlyLine(self, i, norm = False, xmin = 0, xmax = 1):
		line = self.frame[self.frame['flyline'] == i]
		filtLine = line[(line.x > xmin) & (line.x < xmax)]
		if norm:
			const = filtLine.y.max()
			return (filtLine.x, filtLine.y/const)
		else:
			return (filtLine.x, filtLine.y)
	
	def returnMaxValueForFlyLine(self, i):
		line = self.frame[self.frame['flyline'] == i]
		return line.y.max()
	
	def getBinnedDataForFlyLine(self, i, bs = 50, norm = False, xmin = 0, xmax = 1):
		x, y = self.returnPointCloudForFlyLine(i, norm, xmin, xmax)
		x = x.values
		y = y.values
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
		
		return avs, stds		
	
	def plotMeanVsAvForAllLines(self, bs = 50, norm = False):
		colors = cm.rainbow(np.linspace(0, 1, self.numLines))
		
		for i in xrange(self.numLines):
			a, s = self.getBinnedDataForFlyLine(i+1, bs, norm)
			plt.scatter(a, s, color=colors[i], alpha = 0.5)
		plt.show()
		
	def plotAveragesForAllLines(self, bs = 50, norm = False):
		colors = cm.rainbow(np.linspace(0, 1, self.numLines))
		
		for i in xrange(self.numLines):
			a, s = self.getBinnedDataForFlyLine(i+1, bs, norm)
			x = np.linspace(0, 1, len(a))
			plt.errorbar(x, a, yerr = s, color=colors[i], alpha = 0.5)
		plt.show()
		
