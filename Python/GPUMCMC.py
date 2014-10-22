


try:
	import pycuda.driver as drv
	import pycuda.autoinit
	from pycuda import curandom
	from pycuda import gpuarray
	from pycuda.compiler import SourceModule
	# from pycuda.tools import DeviceData

	with open("../Cuda/KernelV2.cu") as f:
		mod = SourceModule(f.read())

	likelihoodKernel = mod.get_function("likelihoodKernel")
	likelihoodKernel.prepare('PPPPiiiP')

	#pycuda setup
except ImportError:
	#Running on non cuda computer
	pass #do other stuff
	print "Import Error for PyCuda"


import numpy as np
from scipy.stats import norm
from emailScripy import alertMe
import cPickle
import time


inputDataLen = 256
numPoints = inputDataLen/2
numRuns = 1000



with open("../FixedDataSet/Mean2,3;{}pts;1dim.txt".format(inputDataLen)) as f:
	Xpoints = cPickle.load(f).astype(np.float32)

try:
	numpoints, dim = Xpoints.shape
except ValueError:
	numpoints = Xpoints.shape[0]
	dim = 1

numMixtures = 2

Xpoints_gpu = gpuarray.to_gpu_async(Xpoints.astype(np.float32))
#quick transfer

#fixed vals
weights_gpu = gpuarray.to_gpu_async(np.array([[0.5], [0.5]]).astype(np.float32))
diagCovs_gpu = gpuarray.to_gpu_async(np.array([[1], [1]]).astype(np.float32))

#output
emptyLikelihood_gpu = gpuarray.zeros(shape = int(5), dtype = np.float32)

#allocate space
means_gpu = gpuarray.zeros(shape = (int(numMixtures), int(dim)), dtype = np.float32)

#local numpy copy
means = np.array([[2],[3]]).astype(np.float32)

#means_gpu.set_async(means)


oldLL = -10000
newLL = 0
samples = np.zeros((numRuns, 2))
diagVal = 1
acceptNum = 0
currentPos = 0

for k in xrange(numRuns):
	# if k%100==0:
	print "At ", k, " iterations"

	proposal = np.random.multivariate_normal(mean = [0,0], cov = np.diag([diagVal, diagVal])).astype(np.float32)

	newMeans = means + proposal.reshape((numMixtures, dim))
	
	means_gpu.set_async(means)

	likelihoodKernel.prepared_call((1,1), (inputDataLen, 1,1),  
	Xpoints_gpu, means_gpu.gpudata, diagCovs_gpu.gpudata, weights_gpu.gpudata, 
	dim, numPoints, numMixtures,	
	emptyLikelihood_gpu.gpudata)



