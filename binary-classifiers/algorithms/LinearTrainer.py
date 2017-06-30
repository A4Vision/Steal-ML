__author__ = 'Fan'

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import sklearn.metrics as sm

from OfflineBase import OfflineBase
from utils.logger import *
from utils.result import Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class LinearTrainer(OfflineBase):
    """
    Trains a linear classifier and returns the result
    (over training, validation and test).
    """
    def __init__(self, name, x, y, val_x, val_y, test_x, test_y, n_features):
        """

        :param name:
        :param x: training data
        :param y: training labels
        :param val_x: validation data
        :param val_y: validation labels
        :param test_x: test data
        :param test_y: test labels
        :param n_features: number of features
        """
        super(self.__class__, self).__init__(x, y, val_x, val_y)

        self.name = name
        self.n_features = n_features
        self.test_x = test_x
        self.test_y = test_y

    def grid_search(self):
        """
        Select the "best" linear classifier, minimizing
            ||w|| ** 2 + C * SUM_i(hinge_loss(x_i, y_i) ** 2)
        Minimize
        :return:
        """
        C_range = np.logspace(-5, 15, 21, base=2)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(self.y_ex, n_iter=5,
                                    test_size=0.2, random_state=42)
        # LinearSVC is an SVM optimizer that minimizes:
        #   ||w|| ** 2 + C * SUM_i(hinge_loss(x_i, y_i) ** 2)
        grid = GridSearchCV(LinearSVC(dual=False, max_iter=10000),
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=1, verbose=0)

        logger.info('start grid search for Linear')
        grid.fit(self.X_ex, self.y_ex)
        logger.info('end grid search for Linear')

        scores = [x[1] for x in grid.grid_scores_]

        # final train
        rbf_svc2 = grid.best_estimator_

        pred_train = rbf_svc2.predict(self.X_ex)
        pred_val = rbf_svc2.predict(self.val_x)
        pred_test = rbf_svc2.predict(self.test_x)

        r = Result(self.name + ' (X)', 'Linear', len(self.X_ex),
                   sm.accuracy_score(self.y_ex, pred_train),
                   sm.accuracy_score(self.val_y, pred_val),
                   sm.accuracy_score(self.test_y, pred_test))
        return r
