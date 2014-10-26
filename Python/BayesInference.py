__author__ = 'Varun Nayyar'

import numpy as np
from matplotlib import pyplot as plt
from RobustLikelihoodClass import Likelihood


def main():
    pass


def BayesProb(utterance):
    """

    :param utterance: np.array
    :return:
    """

    llEval = Likelihood(utterance, numMixtures=8)





if __name__ == "__main__":
    main()