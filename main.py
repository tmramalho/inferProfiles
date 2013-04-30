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
	wd = "final/" + csvName[:-4] + "_" + model
	dataContainer = ExpData(csvName, 50)
	dataContainer.plotData(wd)
	dataContainer.saveData(wd+"/expData.csv")
		
	opt = MaxLikelihood()
	xbest = opt.checkIfExists(csvName, model, False)
	if xbest == None:
		dim = 15
		if model == 'none': dim = 24
		xbest = opt.run(dataContainer, model, dim)
		opt.save(csvName, model)
		
	if model == 'intext':
		ss = SplineSmoother(xbest[:-3], wd, dataContainer.scale, None)
		ss.showSpline()
		ss.plotSplineData(dataContainer, xbest[-2], xbest[-1], xbest[-3], dataContainer.max)
		ss.plotBinnedData(dataContainer, xbest[-2], xbest[-1], xbest[-3], dataContainer.minx, dataContainer.maxx)
		ss.plotFisherInfo(dataContainer, xbest[-2], xbest[-1], xbest[-3], 0.1, 0.005)
		ss.saveSpline(wd+"/spline.csv")
	else:
		ss = DualSplineSmoother(xbest, wd, dataContainer.scale, None)
		ss.showSpline(1)
		ss.plotSplineData(dataContainer, dataContainer.max)
		ss.plotBinnedData(dataContainer)
		ss.plotFisherInfo(dataContainer, 0.1, 0.005)
		ss.saveSpline(wd+"/spline.csv")
		ss.saveSigmaSpline(wd+"/sigmaspline.csv")
		
	