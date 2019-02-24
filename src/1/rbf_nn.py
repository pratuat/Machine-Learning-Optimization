##
import time
import numpy as np
import scipy.optimize as optimize
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn import cross_validation as cval
from sklearn.metrics import confusion_matrix

import pandas as pd
import pdb
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]
train_1.iloc[:, 0] = 1 # (1005, 257)
train_8.iloc[:, 0] = 0 # (542, 257)

# test_1 = pd.read_csv('data/Test_1.csv')
# test_8 = pd.read_csv('data/Test_8.csv')
# test_1.iloc[:, 0] = 0
# test_8.iloc[:, 0] = 1

train_data = pd.concat([train_1.iloc[:, :], train_8.iloc[:, :]], axis=0, ignore_index=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gaussian_function(x, c, sigma):
    return np.exp(-(np.linalg.norm(x - c) / sigma) ** 2)

class RbfNN(BaseEstimator):
    def __init__(self, noc = 5, solver = 'BFGS', sigma = 2, rho1 = 1e-4, rho2 = 1e-4):
        self.noc = noc
        self.solver = solver
        self.sigma = sigma
        self.rho1 = rho1
        self.rho2 = rho2

        self.X = None
        self.Y = None
        self.X_dim = None
        self.result = None
        self.centers = None
        self.weights = None
        self.interpolation_matrix = None

    def __setup_centers(self):
        self.centers = KMeans(
            n_clusters=self.noc,
            random_state=0,
            init='k-means++'
        ).fit(self.X).cluster_centers_

    def __setup_weights(self):
        self.weights = 0.1 * np.random.random(self.noc)

    def __setup_interpolation_matrix(self, x):
        interpolation_matrix = np.zeros((len(x), len(self.centers)))
        for x_idx, vector in enumerate(x):
            for c_idx, center in enumerate(self.centers):
                interpolation_matrix[x_idx, c_idx] = gaussian_function(vector, center, self.sigma)

        return interpolation_matrix

    def __set_centers_and_weights(self, training_parameters):
        self.centers = np.array(training_parameters[:self.noc * self.X_dim]).reshape(self.noc, self.X_dim)
        self.weights = np.array(training_parameters[self.noc * self.X_dim:])

    def __prediction_error(self, py, ty):
        return np.sum(np.square(py - ty)) # / (2 * len(ty))

    def __to_binary(self, x):
        return x > 0.5

    def __print_model_params(self):
        print('='*50)
        print(self.noc, self.solver, self.sigma, self.rho1, self.rho2)
        print('=' * 50)

    def __error_function(self, training_parameters):
        self.__set_centers_and_weights(training_parameters)
        interpolation_matrix = self.__setup_interpolation_matrix(self.X)

        # prediction with appropriate activation function
        predictions = sigmoid(np.dot(interpolation_matrix, self.weights))
        # predictions = np.dot(interpolation_matrix, self.weights)

        # pdb.set_trace()

        squared_error = self.__prediction_error(self.__to_binary(predictions), self.Y)
        # squared_error = np.sum(np.square(predictions - self.Y))

        regularization_error = self.rho1 * np.sum(np.square(self.weights)) + self.rho2 * np.sum(np.square(self.centers.flatten()))

        model_error = squared_error + regularization_error

        self.squared_error = squared_error
        # print("SSE: ", squared_error, "ME: ", model_error)
        # print(confusion_matrix(self.Y, self.__to_binary(predictions)))

        return squared_error

    def fit(self, x, y=None):
        self.__print_model_params()
        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.__setup_centers()
        self.__setup_weights()

        # start_time = time.time()

        result = optimize.minimize(
            fun = self.__error_function,
            x0 = np.concatenate((self.centers.flatten(), self.weights), axis = 0),
            method = self.solver,
            options = {
                'maxfun' : 1
            }
        )

        # end_time = time.time()

        self.result = result
        self.__set_centers_and_weights(result.x)

        return self

    def predict(self, x, y=None):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        return self.__to_binary(predictions)

    # def score(self, x, y=None):
    #     predictions = self.predict(x)
    #
    #     return self.__prediction_error(self.__to_binary(predictions), y)


param_grid = {
    'noc' : [20, 30],
    'solver' : ['BFGS'],
    'sigma' : [2, 3, 4]
}

# param_grid = {
#     'noc' : [20, 30, 50],
#     'solver' : ['BFGS', 'L-BFGS-B', 'Newton-CG'],
#     'sigma' : [2, 3, 4],
#     'rho1' : [1e-4, 1e-5, 1e-6],
#     'rho2' : [1e-4, 1e-5, 1e-6]
# }

X = np.asanyarray(train_data.iloc[:, 1:])
Y = np.asanyarray(train_data.iloc[:, 0])

gs = GridSearchCV(
    RbfNN(),
    param_grid,
    cv=2,
    n_jobs=-1,
    error_score=1,
    scoring='accuracy'
).fit(X, y=Y)

##

# rbf_nn = RbfNN(10).fit(
#     np.asanyarray(train_data.iloc[:, 1:]),
#     np.asanyarray(train_data.iloc[:, 0])
# )

##
