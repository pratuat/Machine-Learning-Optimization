##
import time
import numpy as np
import scipy.optimize as optimize
from sklearn.cluster import KMeans

import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

##

train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]
test_1 = pd.read_csv('data/Test_1.csv')
test_8 = pd.read_csv('data/Test_8.csv')

train_1.iloc[:, 0] = 1
train_8.iloc[:, 0] = 0
test_1.iloc[:, 0] = 1
test_8.iloc[:, 0] = 0

train_data = pd.concat([train_1, train_8], axis=0, ignore_index=True)

##

class RbfNN:
    def __init__(self, noc, sigma = 2, rho1 = 10e-4, rho2 = 10e-4, solver = 'L-BFGS-B'):
        self.noc = noc
        self.sigma = sigma
        self.rho1 = rho1
        self.rho2 = rho2
        self.solver = solver

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
                interpolation_matrix[x_idx, c_idx] = self.gaussian_function(vector, center)

        return interpolation_matrix

    def __set_centers_and_weights(self, training_parameters):
        self.centers = np.array(training_parameters[:self.noc * self.X_dim]).reshape(self.noc, self.X_dim)
        self.weights = np.array(training_parameters[self.noc * self.X_dim:])

    def __error_function(self, training_parameters):
        self.__set_centers_and_weights(training_parameters)
        interpolation_matrix = self.__setup_interpolation_matrix(self.X)

        # prediction with appropriate activation function
        predictions = self.sigmoid(np.dot(interpolation_matrix, self.weights))
        # predictions = np.dot(interpolation_matrix, self.weights)
        squared_error = np.sum(np.square(predictions - self.Y)) / (2 * len(self.Y))
        # squared_error = np.sum(np.square(predictions - self.Y))
        print("Sum of Squared Error: ", squared_error)

        return squared_error

    def fit(self, x, y):
        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.__setup_centers()
        self.__setup_weights()

        start_time = time.time()

        result = optimize.minimize(
            fun = self.__error_function,
            x0 = np.concatenate((self.centers.flatten(), self.weights), axis = 0),
            method = self.solver
            # options = {
            #     'maxiter' : 100
            # }
        )

        end_time = time.time()

        self.result = result
        self.__set_centers_and_weights(result.x)

        return self

    def predict(self, x):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        return predictions

    def gaussian_function(self, x, c):
        return np.exp(-(np.linalg.norm(x - c) / self.sigma) ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

##

rbf_nn = RbfNN(30).fit(
    np.asanyarray(train_data.iloc[:, 1:]),
    np.asanyarray(train_data.iloc[:, 0])
)

##

