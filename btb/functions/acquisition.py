# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm


class AcquisitionFunctionMixin(metaclass=ABCMeta):

    @abstractmethod
    def _acquire(self, predictions, n_candidates):
        pass


class ExpectedImprovementAcquisitionFunction(AcquisitionFunctionMixin):

    def _acquire(self, predictions, n_candidates=1):

        if n_candidates > 1:
            raise NotImplementedError

        Phi = norm.cdf
        N = norm.pdf
        mu, sigma = predictions.T
        y_best = np.max(self.y)
        z = (mu - y_best) / sigma
        ei = sigma * (z * Phi(z) + N(z))
        candidate = np.argmax(ei)
        return [candidate]
