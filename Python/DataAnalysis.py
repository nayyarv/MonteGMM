import cPickle
from matplotlib import pyplot as plt
import numpy as np

print "Starting"

with open("../Data/Mean2,3;64pts;1dim;MCMCRes800000.txt") as f:
	samples  = cPickle.load(f)

print "Finished Pickling"

# print samples
burnIn  = 100
endPoint = 100000
lag  = 70



plt.subplot(211)
plt.acorr(samples.T[0][burnIn:endPoint]-np.mean(samples.T[0][burnIn:endPoint]), maxlags=100)
# pri/nt samples.T[0][burnIn:]

plt.subplot(212)
plt.acorr(samples.T[1][burnIn:endPoint]-np.mean(samples.T[1][burnIn:endPoint]), maxlags=100)
# print samples.T[1][burnIn:]
plt.show()

# import sys
# sys.exit(1)
plt.figure()

plt.hexbin(samples.T[0][burnIn::lag], samples.T[1][burnIn::lag])


plt.show()