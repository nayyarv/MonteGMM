import numpy as np
from scipy.stats import norm

def pythonLL(Xpoints, means, diagCovs, weights):
	numPoints, dim = Xpoints.shape
	numMixtures = len(weights)

	ll = np.zeros(numPoints)

	constMulti = dim/2.0 * np.log(2*np.pi)

	CovDet = np.zeros(numMixtures)
	
	for i in xrange(numMixtures):
		CovDet[i] = 1.0/np.sqrt(np.prod(diagCovs[i]))


	for i in xrange(numPoints):
		for mixes in xrange(numMixtures):
			multiVal = 1
			
			temp = np.dot((Xpoints[i]-means[mixes])/diagCovs[mixes],(Xpoints[i]-means[mixes]))
			temp *= -0.5
			ll[i] += weights[mixes] * np.exp(temp) * CovDet[mixes]
			
		ll[i] = np.log(ll[i]) - constMulti
		

	# print "Now the Log + Likelihood"
	# print np.sum(ll)
	return np.sum(ll)







def pythonLLScipy(Xpoints, means, diagCovs,weights):
	from scipy.stats import multivariate_normal
	
	numPoints, dim = Xpoints.shape
	numMixtures = len(weights)

	ll = np.zeros(numPoints)

	# constMulti = numMixtures/2.0 * np.log(2*np.pi)

	for i in xrange(numPoints):
		for mixes in xrange(numMixtures):
			temp = weights[mixes] * multivariate_normal.pdf(x = Xpoints[i], mean =means[mixes], cov = np.diag(diagCovs[mixes]))
			# print "Temp: ", temp
			ll[i] += temp

	return np.sum(np.log(ll))

def largertest(numRuns = 1000, numPoints = 512, dim = 13, numMixtures = 8):



	Xpoints = np.random.normal(size=(numPoints,dim)).astype(np.float32)
	means = np.random.normal(size=(numMixtures,dim)).astype(np.float32)
	diagCovs = np.random.uniform(size=(numMixtures,dim)).astype(np.float32)
	weights = np.random.uniform(size=numMixtures).astype(np.float32)
	weights/=np.sum(weights)

	for i in xrange(numRuns):
		if i%10==0: print "At {} iterations".format(i)
		means = np.random.normal(size=(numMixtures,dim)).astype(np.float32)
		diagCovs = np.random.uniform(size=(numMixtures,dim)).astype(np.float32)
		weights = np.random.uniform(size=numMixtures).astype(np.float32)
		weights/=np.sum(weights)
	
		# tp  =pythonLLScipy(Xpoints, means, diagCovs, weights)
		tp2 = pythonLL(Xpoints, means, diagCovs, weights)
		# print tp, tp2, tp-tp2

	print "NumRuns: {}, numPoints: {} ".format(numRuns, numPoints)

if __name__ == '__main__':
	import sys
	if len(sys.argv) == 3:
		largertest(numRuns = int(sys.argv[1]), numPoints = int(sys.argv[2]), dim = 13, numMixtures = 8)
		# main(, )
		#We have a input length and numRuns length
	elif len(sys.argv)==2:
		largertest(numRuns = 1000, numPoints = int(sys.argv[1]), dim = 13, numMixtures = 8)
	elif len(sys.argv)==1:
		#run with default
		largertest(numRuns = 1000, numPoints = 64, dim = 13, numMixtures = 8)
	else:
		print "Failure"

	
