#Data generator


import numpy as np
import cPickle


numPoints = 128
dim  = 1
numMixtures = 2

Xpoints1 = np.random.normal(loc = 2, size=(numPoints,dim)).astype(np.float32)
Xpoints2 = np.random.normal(loc = 3, size=(numPoints,dim)).astype(np.float32)

Xpoints = np.vstack((Xpoints1, Xpoints2))
print Xpoints


with open("../FixedDataSet/Mean2,3;{}pts;{}dim.txt".format(numPoints*numMixtures, dim), 'w') as f:
	cPickle.dump(Xpoints, f)


