#Monte Carlo 



import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

with open("KernelV2.cu") as f:
	mod = SourceModule(f.read())

likelihoodKernel = mod.get_function("likelihoodKernel")
likelihoodKernel.prepare('PPPPiiiP')

numPoints = np.uint32(128)
dim = np.uint32(1)
numMixtures = np.uint32(2)

#Generated data!!
Xpoints1 = np.random.normal(loc = 1, size=(numPoints,dim)).astype(np.float32)
Xpoints2 = np.random.normal(loc = 2, size=(numPoints,dim)).astype(np.float32)

Xpoints = np.vstack((Xpoints1, Xpoints2))
weights = np.ones(numMixtures)/(1.0*numMixtures)
weights = weights.astype(float32)
# means = np.arange(numMixtures*dim).reshape((numMixtures,dim))
# means=means.astype(np.float32)

#Initial mean
means = np.random.normal(loc = 1.5, scale = 1, size=(numMixtures,dim)).astype(np.float32)

numRuns = 10000

for i in xrange(numRuns):
	proposalFunc = np.random.multivariate_normal(mean = means, np.diag(1,1))
	


