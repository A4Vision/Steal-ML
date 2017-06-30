__author__ = 'fan'

import numpy as np
from sklearn.metrics import accuracy_score


class HyperSolver:
    """
    Unknown reason for this class, see fit() function.
    TODO: see how many dimensions the inputs X have.
    """
    def __init__(self, p=1, n=-1):
        self.w = None
        self.b = None
        self.POS = p
        self.NEG = n

    def fit(self, X, Y):
        """
        Weird - finds a a hyper-plane that s.t.
            FF is ones vector.
        X * w ~= -FF
        X * w + FF ~= 0
            count(sign(X * w + FF) == sign(yy))
        Then, chooses between (w, b) and (-w, -b)
        s.t. W, b predict Y better based on X.
        :param X:
        :param Y:
        :return:
        """
        m = len(X)
        para = np.matrix(X)
        bb = -1 * np.ones(m).T
        print para, bb

        self.w, _, _, _ = np.linalg.lstsq(para, bb)
        self.b = 1

        yy = self.predict(X)
        score = sum(yy == Y) / float(len(yy))
        print score
        if score < 0.5:
            self.w *= -1
            self.b *= -1

    def predict(self, X):
        """
        Returns sign(X * w + b)
        :param X:
        :return:
        """
        yy = np.inner(X, self.w)
        b = self.b * np.ones(yy.shape)
        d = np.sign(np.inner(X, self.w) + b)
        d[d == 1] = self.POS
        d[d == -1] = self.NEG
        return d

    def get_params(self, deep=True):
        return {}

    def score(self, X, y, sample_weight=None):
        """
        Count how many correct predictions for the given inputs and labels.
        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
