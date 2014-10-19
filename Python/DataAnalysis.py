import cPickle
from matplotlib import pyplot as plt
import numpy as np

with open("../Data/Mean2,3;16pts;1dim;MCMCRes800000.txt") as f:
	samples  = cPickle.load(f)

print "Finished Pickling"

# print samples
burnIn  = 100
endPoint = len(samples)
lag  = 25

# plt.subplot(211)
# plt.acorr(samples.T[0][burnIn:endPoint]-np.mean(samples.T[0][burnIn:endPoint]), maxlags=100)
# # print samples.T[0][burnIn:]

# plt.subplot(212)
# plt.acorr(samples.T[1][burnIn:endPoint]-np.mean(samples.T[1][burnIn:endPoint]), maxlags=100)
# print samples.T[1][burnIn:]

plt.figure()

plt.hexbin(samples.T[0][burnIn:endPoint:lag], samples.T[1][burnIn:endPoint:lag])


plt.show()