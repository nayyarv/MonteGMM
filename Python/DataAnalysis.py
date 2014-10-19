import cPickle
from matplotlib import pyplot as plt
import numpy as np

with open("../Data/Mean2,3;16pts;1dim;MCMCRes.txt") as f:
	samples  = cPickle.load(f)

# print samples
burnIn  = 100
lag  =25

plt.subplot(211)
plt.acorr(samples.T[0][burnIn:]-np.mean(samples.T[0][burnIn:]), maxlags=100)
# print samples.T[0][burnIn:]

plt.subplot(212)
plt.acorr(samples.T[1][burnIn:]-np.mean(samples.T[1][burnIn:]), maxlags=100)
# print samples.T[1][burnIn:]

plt.figure()

plt.hexbin(samples.T[0][burnIn::lag], samples.T[1][burnIn::lag])


plt.show()