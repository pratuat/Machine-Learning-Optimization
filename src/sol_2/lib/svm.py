import cvxopt
import numpy as np
import cvxopt.solvers
from numpy import linalg
from sklearn.base import BaseEstimator
import time

#%%

def kernel(x, y, gamma=0.02):
    return np.exp(- (gamma * linalg.norm(x-y)**2))

class SVM(BaseEstimator):
    def __init__(self, gamma=0.02):
        self.gamma = gamma
        self.alphas = None
        self.support_vectors = None
        self.support_vector_op = None
        self.intercepts = None
        self.solution = None
        self.optimization_time = None

    def fit(self, X, Y):
        print('==' * 25)
        print("Fitting: ", self.gamma)
        print('==' * 25)
        no_of_samples, no_of_features = X.shape

        kernel_dist = np.zeros((no_of_samples, no_of_samples))

        for i in range(no_of_samples):
            for j in range(no_of_samples):
                kernel_dist[i,j] = kernel(X[i], X[j], self.gamma)

        P = cvxopt.matrix(np.outer(Y, Y) * kernel_dist)
        q = cvxopt.matrix(np.ones(no_of_samples) * -1)
        A = cvxopt.matrix(Y.astype('d'), (1, no_of_samples))
        b = cvxopt.matrix(0.0)

        G = cvxopt.matrix(np.diag(np.ones(no_of_samples) * -1))
        h = cvxopt.matrix(np.zeros(no_of_samples))

        start = time.time()
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        end = time.time()

        self.optimization_time = end - start

        self.solution = solution

        alphas = np.ravel(solution['x'])
        support_vectors = alphas > 0
        indexes = np.arange(len(alphas))[support_vectors]

        self.alphas = alphas[support_vectors]
        self.support_vectors = X[support_vectors]
        self.support_vector_op = Y[support_vectors]

        self.intercepts = 0

        for n in range(len(self.alphas)):
            self.intercepts += self.support_vector_op[n]
            self.intercepts -= np.sum(self.alphas * self.support_vector_op * kernel_dist[indexes[n], support_vectors])
        self.intercepts /= len(self.alphas)

        return self

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alphas, self.support_vector_op, self.support_vectors):
                s += a * sv_y * kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + self.intercepts

    def predict(self, x, y=None):
        predictions = np.sign(self.project(x))

        return predictions

    def statistics(self):
        return [
            ('Numbers to classify:', '1 and 8'),
            ('Optimization time:', self.optimization_time),
            ('Value of γ:', self.gamma),
            ('Optimization solver chosen:', 'cvxopt'),
            ('Total number of iterations:', self.solution['iterations'])
        ]

#%%
