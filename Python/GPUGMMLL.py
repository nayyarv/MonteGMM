import pycuda.autoinit
# from pycuda import curandom
from pycuda import gpuarray
from pycuda.compiler import SourceModule
# from pycuda.tools import DeviceData
import numpy as np
from pythonGMMLL import pythonLL

#prepare for global usage


def largertest(numRuns = 1000, numPoints = 512, dim = 13, numMixtures = 8):



	with open("../Cuda/KernelV2.cu") as f:
		if numPoints>=1024:
			mod = SourceModule(f.read().replace('512', '1024'))
			numThreads = 1024
		else:
			mod = SourceModule(f.read())
			numThreads = 512

	if numPoints>numThreads:
		numBlocks = numPoints/numThreads
		if numPoints%numThreads != 0: numBlocks+=1
	else:
		numBlocks=1

	print "numBlocks: {}, numPoints: {}".format(numBlocks, numPoints)

	likelihoodKernel = mod.get_function("likelihoodKernel")
	likelihoodKernel.prepare('PPPPiiiP')

	Xpoints = np.random.normal(size=(numPoints,dim)).astype(np.float32)
	means = np.random.normal(size=(numMixtures,dim)).astype(np.float32)
	diagCovs = np.random.uniform(size=(numMixtures,dim)).astype(np.float32)
	weights = np.random.uniform(size=numMixtures).astype(np.float32)
	weights/=np.sum(weights)



	Xpoints_gpu = gpuarray.to_gpu_async(Xpoints)
	diagCovs_gpu = gpuarray.to_gpu_async(diagCovs)
	means_gpu = gpuarray.to_gpu_async(means)
	weights_gpu = gpuarray.to_gpu_async(weights)
	emptyLikelihood_gpu = gpuarray.zeros(shape = int(1), dtype = np.float32)



	for i in xrange(numRuns):
		if i%10==0: 
			print "At {} iterations".format(i)
			
		likelihoodKernel.prepared_call((numBlocks,1), (numThreads, 1,1),  
		Xpoints_gpu.gpudata, means_gpu.gpudata, diagCovs_gpu.gpudata, weights_gpu.gpudata, 
		dim, numPoints, numMixtures,	
		emptyLikelihood_gpu.gpudata)
		ll = emptyLikelihood_gpu.get()[0]

		print ll

	tp =  pythonLL(Xpoints, means, diagCovs, weights)
	print "Correct value: ", tp


if __name__ == '__main__':
	largertest(numRuns = 1000, numPoints = 2048, dim = 13, numMixtures = 8)

