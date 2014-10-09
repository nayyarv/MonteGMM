import numpy as np
from scipy.stats import norm

numPoints = 100
dim = 5
numMixtures = 4


Xpoints = np.random.normal(size=(numPoints,dim)).astype(np.float32)
means = np.random.normal(size=(numMixtures,dim)).astype(np.float)
diagCovs = np.random.uniform(size=(numMixtures,dim)).astype(np.float)
weights = np.random.uniform(size=numMixtures).astype(np.float32)
weights/=np.sum(weights)


ll = np.zeros(numPoints)

for i in xrange(numPoints):
	for mixes in xrange(numMixtures):
		multiVal = 1
		for d in xrange(dim):
			multiVal *= norm.pdf(Xpoints[i][d], loc=means[mixes][d], scale=diagCovs[mixes][d])
		
		ll[i] += weights[mixes] * multiVal


print "Now the Log + Likelihood"
print np.sum(np.log(ll))


