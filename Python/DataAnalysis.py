import cPickle
from matplotlib import pyplot as plt
import numpy as np

print "Starting"

numPoints = 256

with open("../Data/Mean2,3_{}pts_1dim_MCMCRes800000.txt".format(numPoints)) as f:
	samples  = cPickle.load(f)

print "Finished Pickling"

# print samples
burnIn  = 1000
endPoint = 100000
lag  = 50

acorrOnly = False

if acorrOnly:

	plt.subplot(211)
	plt.acorr(samples.T[0][burnIn:endPoint]-np.mean(samples.T[0][burnIn:endPoint]), maxlags=100)
	# pri/nt samples.T[0][burnIn:]

	plt.subplot(212)
	plt.acorr(samples.T[1][burnIn:endPoint]-np.mean(samples.T[1][burnIn:endPoint]), maxlags=100)
	# print samples.T[1][burnIn:]
	plt.show()

else:
	plt.figure(tight_layout=True)
	plt.title("log $p(\\theta|x)$, {} points".format(numPoints))
	plt.xlabel("$\mu_1$")
	plt.ylabel("$\mu_2$")
	# plt.xlim((1,3.5))
	# plt.ylim((1,3.5))
	plt.hexbin(samples.T[0][burnIn::lag], samples.T[1][burnIn::lag], bins = 'log', cmap=plt.cm.Greys)


	plt.show()