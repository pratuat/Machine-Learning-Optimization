import numpy as np
import scipy.optimize as optimize
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


class RbfNNBC(BaseEstimator):
    def __init__(self, noc = 5, solver = 'BFGS', sigma = 2, rho = 1e-3, optimizer_options = {}):
        self.noc = noc
        self.solver = solver
        self.sigma = sigma
        self.rho = rho
        self.optimizer_options = optimizer_options

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
        return (x > 0.5).astype(int)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __setup_interpolation_matrix(self, x):
        return rbf_kernel(X=x, Y=self.centers, gamma=1/self.sigma**2)

    def __sum_of_squared_error(self, py, ty):
        return np.sum(np.square(py - ty)) / (2 *len(ty))

    def __print_model_params(self, tag):
        print('='*50)
        print(tag, "NOC: ", self.noc, "SOLVER: ", self.solver, "SIGMA: ", self.sigma, "RHO: ", self.rho)
        print('=' * 50)

    def __error_function(self, training_parameters):
        self.__set_centers_and_weights(training_parameters)

        interpolation_matrix = self.__setup_interpolation_matrix(self.X)

        predictions = np.dot(interpolation_matrix, self.weights)

        sum_of_squared_error = self.__sum_of_squared_error(predictions, self.Y)
        regularization_error = self.rho/2 * (np.sum(np.square(self.weights)) + np.sum(np.square(self.centers.flatten())))
        total_error = sum_of_squared_error + regularization_error

        return total_error

    def fit(self, x, y=None):
        self.__print_model_params('[BEGIN]')

        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.__setup_centers()
        self.__setup_weights()

        result = optimize.minimize(
            fun = self.__error_function,
            x0 = np.concatenate((self.centers.flatten(), self.weights), axis = 0),
            method = self.solver,
            options = self.optimizer_options
        )

        self.__print_model_params('[END]')

        self.result = result
        self.__set_centers_and_weights(result.x)

        return self

    def predict(self, x, y=None):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        return self.__to_binary(predictions)

