import time
import numpy as np
import scipy.optimize as optimize
from scipy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


class Rbf(BaseEstimator):
    def __init__(self, noc = 5, solver = 'L-BFGS-B', sigma = 2, rho = 1e-3, optimizer_options = {}):
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

        self.train_squared_error = []
        self.train_total_error = []
        self.train_reg_error = []
        self.prev_training_params = None
        self.current_training_params = None
        self.optimization_time = None
        self.test_error = None

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
        message = " ".join(str(el) for el in (tag, "|", "NOC: ", self.noc, "|", "SOLVER: ", self.solver, "|", "SIGMA: ", self.sigma, "|", "RHO: ", self.rho))

        print('='*50)
        print(message)
        print('=' * 50)

    def __final_gradient_norm(self):
        return norm(self.current_training_params - self.prev_training_params)

    def __error_function(self, training_parameters):
        self.__set_centers_and_weights(training_parameters)

        interpolation_matrix = self.__setup_interpolation_matrix(self.X)

        predictions = np.dot(interpolation_matrix, self.weights)

        sum_of_squared_error = self.__sum_of_squared_error(predictions, self.Y)
        regularization_error = self.rho/2 * (np.sum(np.square(self.weights)) + np.sum(np.square(self.centers.flatten())))
        total_error = sum_of_squared_error + regularization_error

        self.train_squared_error.append(sum_of_squared_error)
        self.train_reg_error.append(regularization_error)
        self.train_total_error.append(total_error)

        self.prev_training_params = self.current_training_params
        self.current_training_params = training_parameters

        return total_error

    def fit(self, x, y):
        # self.__print_model_params('[BEGIN]')

        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.__setup_centers()
        self.__setup_weights()

        start = time.time()
        result = optimize.minimize(
            fun = self.__error_function,
            x0 = np.concatenate((self.centers.flatten(), self.weights), axis = 0),
            method = self.solver,
            options = self.optimizer_options
        )
        end = time.time()
        self.optimization_time = end - start

        self.result = result
        self.__set_centers_and_weights(result.x)

        # self.__print_model_params('[END]')

        return self

    def predict(self, x):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        return self.__to_binary(predictions)

    def test(self, x, y):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        self.test_error = self.__sum_of_squared_error(predictions, y)

        return self.__to_binary(predictions), self.test_error

    def statistics(self):
        return [
            ('Number of neurons N:', self.noc),
            ('Initial Training Error:', self.train_squared_error[0]),
            ('Final Training Error:', self.train_squared_error[-1]),
            ('Final Test Error:', self.test_error),
            ('Norm of the gradient at the final point:', self.__final_gradient_norm()),
            ('Optimization solver chosen:', self.solver),
            ('Total Number of function/gradient evaluations:', self.result.nfev),
            ('Time for optimizing the network (seconds):', self.optimization_time ),
            ('Value of σ:',self.sigma),
            ('Value of ρ:', self.rho)
        ]