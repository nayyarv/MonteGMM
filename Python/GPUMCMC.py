


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
numRuns = 800000

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
diagCovs_gpu = gpuarray.to_gpu_async(np.array([[1],[1]]).astype(np.float32))
weights_gpu = gpuarray.to_gpu_async(np.array([[0.5,0.5]]).astype(np.float32))

#output
emptyLikelihood_gpu = gpuarray.zeros(shape = int(1), dtype = np.float32)

#allocate space
means_gpu = gpuarray.zeros(shape = (int(numMixtures), int(dim)), dtype = np.float32)

#local numpy copy
means = np.array([[2],[3]]).astype(np.float32)

#means_gpu.set_async(means)


oldLL = -1000
newLL = 0
samples = np.zeros((numRuns, 2))
diagVal = 0.8
acceptNum = 0
currentPos = 0

diagNear = 0.2
diagFar = 1

covFar = np.diag([diagFar,diagFar]).astype(np.float32)
covNear = np.diag([diagNear,diagNear]).astype(np.float32)

for k in xrange(numRuns):
	if k%100==0:
		print "At ", k, " iterations" 
	
	if currentPos>1:
		covMat = covNear
	else:
		covMat = covFar

	proposal = np.random.multivariate_normal(mean = [0,0], cov = covMat).astype(np.float32)

	newMeans = means + proposal.reshape((numMixtures, dim))
	
	means_gpu.set(newMeans)#No Async on this one!

	likelihoodKernel.prepared_call((1,1), (numpoints, 1,1),  
	Xpoints_gpu.gpudata, means_gpu.gpudata, diagCovs_gpu.gpudata, weights_gpu.gpudata, 
	dim, numpoints, numMixtures,	
	emptyLikelihood_gpu.gpudata)

	newLL = emptyLikelihood_gpu.get()[0]

	acceptProb = np.exp(newLL-oldLL)

	# print k, means.T[0], newMeans.T[0], 

	if ( acceptProb>=1 or acceptProb>np.random.uniform()):
		means = newMeans
		oldLL = newLL
		acceptNum+=1
		currentPos = 0
		# print " A ", 
	else:
		currentPos+=1
		# print  " R ", 

	# print newLL, acceptProb

	samples[k] = (means.T[0]+0)

print 1.0*acceptNum/numRuns

with open("../Data/Mean2,3_{}pts_1dim_MCMCRes{}.txt".format(inputDataLen, numRuns), 'w') as f:
	cPickle.dump(samples, f)


