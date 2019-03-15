import numpy as np
import scipy.optimize as optimize
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import norm
import time


class RbfBD(BaseEstimator):
    def __init__(self, noc = 5, solver = 'L-BFGS-B', sigma = 2, rho=1e-3,
                 max_iter = None, epsilon = 1e-6, optimizer_options = {}):
        self.noc = noc
        self.solver = solver
        self.sigma = sigma
        self.rho = rho
        self.max_iter = max_iter
        self.epsilon = epsilon
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

    def __set_centers_and_weights(self, training_parameters, mode = 0):
        if mode == 1:
            self.centers = training_parameters.reshape(self.noc, self.X_dim)
        elif mode == 2:
            self.weights = training_parameters
        else:
            self.centers = np.array(training_parameters[:self.noc * self.X_dim]).reshape(self.noc, self.X_dim)
            self.weights = np.array(training_parameters[self.noc * self.X_dim:])

    def __to_binary(self, x):
        return x > 0.5

    def __print_model_params(self):
        print('='*50)
        print(self.noc, self.solver, self.sigma, self.rho)
        print('=' * 50)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sum_of_squared_error(self, py, ty):
        return np.sum(np.square(py - ty)) / (2 *len(ty))

    def __setup_interpolation_matrix(self, x):
        self.interpolation_matrix = rbf_kernel(X=x, Y=self.centers, gamma=1/self.sigma**2)
        return self.interpolation_matrix

    def __final_gradient_norm(self):
        return norm(self.current_training_params - self.prev_training_params)

    def __error_function(self, training_parameters, obj_func_args):
        # pdb.set_trace()
        self.__set_centers_and_weights(training_parameters, mode = obj_func_args)

        if obj_func_args != 2:
            self.__setup_interpolation_matrix(self.X)

        predictions = np.dot(self.interpolation_matrix, self.weights)

        sum_of_squared_error = self.__sum_of_squared_error(predictions, self.Y)
        regularization_error = self.rho/2 *(np.sum(np.square(self.weights)) + np.sum(np.square(self.centers.flatten())))
        total_error = sum_of_squared_error + regularization_error

        self.train_squared_error.append(sum_of_squared_error)
        self.train_reg_error.append(regularization_error)
        self.train_total_error.append(total_error)

        self.prev_training_params = self.current_training_params
        self.current_training_params = training_parameters

        return total_error

    def fit(self, x, y = None):
        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.__setup_centers()
        self.__setup_weights()

        train_accuracy = 0
        iteration = 0

        start = time.time()
        while True:
            print("Iteration: ", iteration)

            # print('INFO: Minimizing centers.')
            result_1 = optimize.minimize(
                fun = self.__error_function,
                x0 = self.centers.flatten(),
                args = 1,
                method = self.solver,
                options = self.optimizer_options
            )

            self.__set_centers_and_weights(result_1.x, mode = 1)

            # print('INFO: Minimizing weights.')
            self.__setup_interpolation_matrix(self.X)
            result_2 = optimize.minimize(
                fun = self.__error_function,
                x0 = self.weights,
                args = 2,
                method = self.solver,
                options = self.optimizer_options
            )

            self.__set_centers_and_weights(result_2.x, mode = 2)

            iteration += 1

            if self.max_iter is not None and iteration >= self.max_iter:
                break

            accuracy = self.__train_accuracy()

            print("Train Accuracy: ", accuracy)

            if abs(accuracy - train_accuracy) < self.epsilon:
                break
            else:
                 train_accuracy = accuracy

        end = time.time()
        self.optimization_time = end - start

        self.result = (result_1, result_2)

        return self

    def __train_accuracy(self):
        train_predictions = self.predict(self.X)

        return 1 - np.sum(np.square(self.Y - train_predictions))/len(self.Y)

    def predict(self, x, y=None):
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
            ('Total Number of function/gradient evaluations:', len(self.train_squared_error)),
            ('Time for optimizing the network (seconds):', self.optimization_time ),
            ('Value of σ:',self.sigma),
            ('Value of ρ:', self.rho)
        ]
