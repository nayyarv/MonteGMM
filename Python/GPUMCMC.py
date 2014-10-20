import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
#pycuda setup

import numpy as np
from scipy.stats import norm
from emailScripy import alertMe
import cPickle
import time


with open("KernelV2.cu") as f:
	mod = SourceModule(f.read())

likelihoodKernel = mod.get_function("likelihoodKernel")
likelihoodKernel.prepare('PPPPiiiP')

with open("../FixedDataSet/Mean2,3;{}pts;1dim.txt".format(inputDataLen)) as f:
	Xpoints = cPickle.load(f).astype(np.float32)

try:
	numpoints, dim = Xpoints.shape
except ValueError:
	numpoints = Xpoints.shape[0]
	dim = 1

numMixtures = 2

Xpoints_gpu = gpuArray.to_gpu_async(Xpoints)




