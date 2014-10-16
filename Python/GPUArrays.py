import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray
from pycuda import curandom

import numpy as np

with open("../Cuda/KernelV2.cu") as f:
	mod = SourceModule(f.read())




numPoints = np.int32(128)
dim = np.int32(5)
numMixtures = np.int32(4)

#Generated data!!
Xpoints = np.random.normal(size=(numPoints,dim)).astype(np.float32)

# means = np.random.normal(size=(numMixtures,dim)).astype(np.float32)
# means = np.arange(numMixtures*dim).reshape((numMixtures,dim))
# means=means.astype(np.float32)

diagCovs = np.random.uniform(size=(numMixtures,dim)).astype(np.float32)+1
weights = np.random.uniform(size=numMixtures).astype(np.float32)
weights/=np.sum(weights)

emptyLikelihood = np.zeros(numPoints).astype(np.float32)

Xpoints_gpu = drv.mem_alloc(Xpoints.nbytes)
# means_gpu = drv.mem_alloc(means.nbytes)
diagCovs_gpu = drv.mem_alloc(diagCovs.nbytes)
weights_gpu = drv.mem_alloc(weights.nbytes)
emptyLikelihood_gpu = drv.mem_alloc(emptyLikelihood.nbytes)

drv.memcpy_htod(Xpoints_gpu,Xpoints)
# drv.memcpy_htod(means_gpu, means)
drv.memcpy_htod(diagCovs_gpu, diagCovs)
drv.memcpy_htod(weights_gpu, weights)



numGen = curandom.MRG32k3aRandomNumberGenerator()
means_gpu = numGen.gen_normal(shape=(numPoints,dim), dtype = np.float32)

drv.memcpy_dtoh(means, means_gpu)

drv.memcpy_htod(emptyLikelihood_gpu, emptyLikelihood)

likelihoodKernel = mod.get_function("likelihoodKernel")

likelihoodKernel(Xpoints_gpu, means_gpu, diagCovs_gpu, weights_gpu, 
	dim, numPoints, numMixtures, 
	emptyLikelihood_gpu,
	block = (128,1,1))

# diag_kernel(, a_stride, a_N, block = (blcksize,1,1))

drv.memcpy_dtoh(emptyLikelihood, emptyLikelihood_gpu)


print emptyLikelihood[0]
from scipy.stats import norm

ll = np.zeros(numPoints)

for i in xrange(numPoints):
	for mixes in xrange(numMixtures):
		multiVal = 1
		for d in xrange(dim):
			x = Xpoints[i][d]
			m  =means[mixes][d]
			c = diagCovs[mixes][d]

			# multiVal*= 1/np.sqrt(2*np.pi*c)*np.exp(-1.0/2 * ((x-m)**2)/c)
			multiVal *= norm.pdf(Xpoints[i][d], loc=means[mixes][d], scale=np.sqrt(diagCovs[mixes][d]))
		
		ll[i] += weights[mixes] * multiVal


# print np.log(ll)


print "Now the Log + Likelihood"
print np.sum(np.log(ll))




