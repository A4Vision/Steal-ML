from regression_stealer import *
from sklearn.linear_model import LogisticRegression
from collections import Counter
import argparse
import sys
import matplotlib.pyplot as plt


class LocalRegressionExtractor(RegressionExtractor):
    """
    Local logistic regression using the implementation in scikit.

    The constructor creates the oracle.
    """

    def __init__(self, X, y, multinomial, rounding=None):
        self.classes = y.unique()
        self.features = X.columns.values
        self.rounding = rounding

        # train a model on the whole dataset
        # self.model is the oracle.
        if multinomial:
            # sovler: the optimization method.
            self.model = LogisticRegression(multi_class="multinomial",
                                            solver='lbfgs')
        else:
            self.model = LogisticRegression(multi_class="ovr")
        self.model.fit(X, y)

        self.w = self.model.coef_
        self.intercept = self.model.intercept_
        self.multinomial = multinomial

        RegressionExtractor.__init__(self)

    def num_features(self):
        return len(self.features)

    def get_classes(self):
        return self.classes

    def query_probas(self, X):
        """
        Call to the oracle on X.
        :param X:
        :return:
        """
        #
        # There seems to be a bug in the LogisticRegression class, that makes
        # it use the OvR strategy to compute probabilities even when we set
        # 'multi_class = multinomial'. So we call the predict_probas method
        # ourselves.

        # New comment: Probably there is a BUG here because LogisticRegression.predict_proba() calculates softmax well.
        p = predict_probas(X, self.w, self.intercept,
                           multinomial=(self.model.multi_class == "multinomial")
                           )

        if self.rounding is None:
            return p
        else:
            p = np.round(p, self.rounding)
            # Re-normalize - make sure each row is a probability vector.
            return p / np.sum(p, axis=1)[:, np.newaxis]

    def query(self, X):
        return predict_classes(X, self.w, self.intercept, self.classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('--multinomial', dest='multinomial',
                        action='store_true', help='multinomial softmax flag')
    parser.add_argument('--rounding', type=int, help='rounding digits')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    data = args.data
    seed = args.seed

    print >> sys.stderr, 'Data: {}, Seed: {}'.format(data, seed)

    np.random.seed(0)
    X_train, y_train, X_test, y_test, _ = utils.prepare_data(data)

    ext = LocalRegressionExtractor(X_train, y_train,
                                   multinomial=args.multinomial,
                                   rounding=args.rounding)

    y_pred = ext.model.predict(X_test)
    print 'training accuracy: {}'.format(accuracy_score(y_test, y_pred))
    print Counter(y_pred)

    ext.run(data, X_test, random_seed=seed,
            #alphas=[1],
            methods=['passive'], baseline=True
            )

    
if __name__ == "__main__":
    main()
