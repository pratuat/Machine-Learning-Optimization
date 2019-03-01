##
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import rbf_kernel

# import pickle
# from numba import vectorize, jit
# from sklearn.model_selection import GridSearchCV

##

data = pd.read_csv('data/DATA.csv')
data_split_margin = int(0.8 * data.shape[0])
X_train_data = np.asanyarray(data.iloc[:data_split_margin, 0:2])
Y_train_data = np.asanyarray(data.iloc[:data_split_margin, 2])
X_test_data = np.asanyarray(data.iloc[data_split_margin:, 0:2])
Y_test_data = np.asanyarray(data.iloc[data_split_margin:, 2])

##
class RBF(BaseEstimator):
    def __init__(self, noc = 5, solver = 'L-BFGS-B', sigma = 2, rho1 = 1e-3, rho2 = 1e-3):
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

    def __set_centers_and_weights(self, training_parameters):
        self.centers = np.array(training_parameters[:self.noc * self.X_dim]).reshape(self.noc, self.X_dim)
        self.weights = np.array(training_parameters[self.noc * self.X_dim:])

    def __to_binary(self, x):
        return x > 0.5

    def __print_model_params(self):
        print('='*50)
        print(self.noc, self.solver, self.sigma, self.rho1, self.rho2)
        print('=' * 50)

    def __prediction_error(self, py, ty):
        return np.sum(np.square(py - ty)) / len(ty)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __setup_interpolation_matrix(self, x):
        return rbf_kernel(X=x, Y=self.centers, gamma=0.25)

    def __error_function(self, training_parameters):
        self.__set_centers_and_weights(training_parameters)
        interpolation_matrix = self.__setup_interpolation_matrix(self.X)

        predictions = np.dot(interpolation_matrix, self.weights)
        # predictions = np.dot(interpolation_matrix, self.weights)

        # squared_error = self.__prediction_error(self.__to_binary(predictions), self.Y)
        squared_error = np.sum(np.square(predictions - self.Y)) / (2 * len(self.Y))

        # regularization_error = self.rho1/2 * np.sum(np.square(self.weights)) + self.rho2/2 * np.sum(np.square(self.centers.flatten()))

        # model_error = squared_error + regularization_error

        # self.squared_error = squared_error
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

        # model_error = self.__error_function(np.concatenate((self.centers.flatten(), self.weights), axis = 0))

        result = optimize.minimize(
            fun = self.__error_function,
            x0 = np.concatenate((self.centers.flatten(), self.weights), axis = 0),
            method = self.solver
        )

        self.result = result
        self.__set_centers_and_weights(result.x)

        return self

    def predict(self, x, y=None):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        return predictions
##

model = RBF(noc = 30).fit(x = X_train_data, y = Y_train_data)

##

X = np.arange(-2, 2.5, 0.05)
Y = np.arange(-2, 2.5, 0.05)
X, Y = np.meshgrid(X, Y)

inputs = np.array([X.flatten(), Y.flatten()]).T
predicted_outputs = model.predict(inputs)
Z = predicted_outputs.reshape((len(X), len(Y)))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

###

