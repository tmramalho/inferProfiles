'''
Created on Feb 2, 2012

@author: tiago
'''
from ExpData import ExpData
from MaxLikelihood import MaxLikelihood
from SplineSmoother import SplineSmoother
from DualSplineSmoother import DualSplineSmoother

if __name__ == '__main__':
	csvName = 'gregor.csv'
	model = 'none'
	dataContainer = ExpData(csvName, 50)
	dataContainer.plotData()
	
	opt = MaxLikelihood()
	xbest = opt.checkIfExists(csvName, model, True)
	if xbest == None:
		xbest = opt.run(dataContainer, model, 24)
		opt.save(csvName, model)
		
	if model == 'intext':
		ss = SplineSmoother(xbest[:-3], dataContainer.sm)
		ss.showSpline()
		ss.plotSplineData(dataContainer, xbest[-2], xbest[-1], xbest[-3], dataContainer.max)
		ss.plotBinnedData(dataContainer, xbest[-2], xbest[-1], xbest[-3], dataContainer.minx, dataContainer.maxx)
		ss.plotFisherInfo(dataContainer, xbest[-2], xbest[-1], xbest[-3], 0.02)
		ss.saveSpline("results/spline.csv")
	else:
		ss = DualSplineSmoother(xbest, dataContainer.scale, None)
		ss.showSpline(1)
		ss.plotSplineData(dataContainer, dataContainer.max)
		ss.plotBinnedData(dataContainer)
		ss.plotFisherInfo(dataContainer, 5)
		ss.saveSpline("results/spline.csv")
		ss.saveSigmaSpline("results/sigmaspline.csv")
		
	