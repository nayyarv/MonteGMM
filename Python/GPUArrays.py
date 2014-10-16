import pycuda.driver as drv
import pycuda.autoinit
from pycuda import curandom
from pycuda import gpuarray
from pycuda.compiler import SourceModule


import numpy as np

with open("../Cuda/KernelV2.cu") as f:
	mod = SourceModule(f.read())


numPoints = np.int32(128)
dim = np.int32(5)
numMixtures = np.int32(4)

#Generated data!!
Xpoints = np.random.normal(size=(numPoints,dim)).astype(np.float32)
Xpoints_gpu = drv.mem_alloc(Xpoints.nbytes)
drv.memcpy_htod(Xpoints_gpu,Xpoints)


numGen = curandom.MRG32k3aRandomNumberGenerator()

means_gpu = numGen.gen_normal(shape=(int(numMixtures),int(dim)), dtype = np.float32)
diagCovs_gpu = numGen.gen_uniform(shape=(int(numMixtures),int(dim)), dtype = np.float32)+1

weights_gpu = numGen.gen_uniform(shape=(int(numMixtures),int(dim)), dtype = np.float32)
print type(gpuarray.sum(weights_gpu))
# weights_gpu=  

emptyLikelihood_gpu = gpuarray.zeros(shape = int(numPoints))

likelihoodKernel = mod.get_function("likelihoodKernel")
# likelihoodKernel.prepare('PPPPiiiP')

likelihoodKernel(Xpoints_gpu, means_gpu, diagCovs_gpu, weights_gpu, 
	dim, numPoints, numMixtures, 
	emptyLikelihood_gpu,
	block = (128,1,1))

# likelihoodKernel.prepared_call(grid = (1,1), block=(128, 1,1),  Xpoints_gpu, means_gpu, diagCovs_gpu, weights_gpu, dim, numPoints, numMixtures,	emptyLikelihood_gpu)




means = means_gpu.get_async()
weights = weights_gpu.get_async()
diagCovs = diagCovs_gpu.get_async()
emptyLikelihood = emptyLikelihood_gpu.get_async()


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




