__author__ = 'Varun Nayyar'

import numpy as np
from MFCCArrayGen import SadCorpus
from sklearn.mixture import GMM
from RobustLikelihoodClass import Likelihood

def main(numMixtures = 8):
    Xpoints = SadCorpus()
    numPoints, dim = Xpoints.shape

    llEval = Likelihood(Xpoints, numMixtures)

    Model = GMM(8)
    Model.fit(Xpoints)

    print np.sort(Model.weights_)

    print llEval.loglikelihood(Model.means_, Model.covars_, Model.weights_)
    print np.sum(Model.score(Xpoints))

    ##above two are the same!!


if __name__ == "__main__":
    main()