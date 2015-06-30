__author__ = 'varunnayyar'


import numpy as np
from scipy.stats import norm

from RobustLikelihoodClass import Likelihood
from Utils.MFCCArrayGen import SadCorpus


class MCMC(object):

    def __init__(self, dataset = None):
        """
        Parameters
        ----------
        dataset: array_like, (size, num_features)
            The MFCC dataset to evaluate. It defaults to the LDC Sad Corpus if no argument is provided

        """
        if dataset is None:
            self.dataset = SadCorpus()
        self.dataset = dataset

    def


