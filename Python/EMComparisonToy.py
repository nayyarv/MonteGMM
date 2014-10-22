

# from matplotlib import pyplot as plt
# import numpy as numpy

import numpy as np
from sklearn.mixture import GMM
from matplotlib import pyplot as plt
import cPickle

numPoints = 16
with open("../FixedDataSet/Mean2,3;{}pts;1dim.txt".format(numPoints)) as f:
	Xpoints = cPickle.load(f)

def main():

	test = GMM(2, params = 'm', init_params='')
	test.weights_ = np.array([0.5, 0.5])
	test.covars_ = np.array([[1],[1]])
	test.means_ =    np.random.normal(loc = 2.5, size=(2,1))


	test.fit(Xpoints)

	return test.means_
	
if __name__ == '__main__':
	numRuns=100
	means = np.zeros((numRuns, 2))
	for i in xrange(numRuns):
		means[i] = np.array(main().T)
	
	print means

	plt.figure(tight_layout=True)
	plt.title("100 different Frequentist Mean estimation, {} pts".format(numPoints))
	plt.xlabel("$\mu_1$")
	plt.ylabel("$\mu_2$")
	plt.plot(means.T[0], means.T[1],'o')
	plt.show()

