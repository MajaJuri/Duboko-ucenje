from sklearn import svm
import numpy as np
import data
import matplotlib.pyplot as plt


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, kernel='rbf')
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.decision_function(X)

    def support(self):
        return self.clf.support_


if __name__ == "__main__":
    np.random.seed(100)
    C = 2
    K = 6
    N = 10
    param_svm_c = 1

    X, Y_true = data.sample_gmm_2d(K=K, C=C, N=N)

    model_svm = KSVMWrap(X, Y_true, param_svm_c=param_svm_c, param_svm_gamma='auto')
    Y_pred = model_svm.predict(X=X)
    support = model_svm.support()

    data.print_performance_indicators(Y_pred, Y_true)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(function = model_svm.predict, rect=rect)
    #data.graph_surface(function = model_svm.get_scores, rect=rect)
    data.graph_data(X, Y_true, Y_pred, special=support)
    plt.show()

