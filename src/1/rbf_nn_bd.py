##
import pdb
import pickle
import datetime
import numpy as np
import pandas as pd

import scipy.optimize as optimize

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV

##

train_1 = pd.read_csv('/Users/pratuat/course_materials/omml/project/data/Train_1.csv').iloc[:, :257]
train_8 = pd.read_csv('/Users/pratuat/course_materials/omml/project/data/Train_8.csv').iloc[:, :257]
train_1.iloc[:, 0] = 0 # (1005, 257)
train_8.iloc[:, 0] = 1 # (542, 257)

train_data = np.asanyarray(pd.concat([train_1.iloc[:, :], train_8.iloc[:, :]], axis=0, ignore_index=True))
np.random.shuffle(train_data)

train_x_data = normalize(train_data[:, 1:])
train_y_data = train_data[:, 0]

##

class RbfNN(BaseEstimator):
    def __init__(self, noc = 5, solver = 'L-BFGS', sigma = 2, rho1 = 1e-3, rho2 = 1e-3):
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
        print(self.noc, self.solver, self.sigma, self.rho1, self.rho2)
        print('=' * 50)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sum_of_squared_error(self, py, ty):
        return np.sum(np.square(py - ty)) / (2 *len(ty))

    def __setup_interpolation_matrix(self, x):
        self.interpolation_matrix = rbf_kernel(X=x, Y=self.centers, gamma=1/self.sigma**2)
        return self.interpolation_matrix

    def __error_function(self, training_parameters, obj_func_args):
        # pdb.set_trace()
        self.__set_centers_and_weights(training_parameters, mode = obj_func_args)

        if obj_func_args != 2:
            self.__setup_interpolation_matrix(self.X)

        predictions = np.dot(self.interpolation_matrix, self.weights)

        sum_of_squared_error = self.__sum_of_squared_error(predictions, self.Y)
        regularization_error = self.rho1/2 * np.sum(np.square(self.weights)) + self.rho2/2 * np.sum(np.square(self.centers.flatten()))
        total_error = sum_of_squared_error + regularization_error

        print("SSE: ", sum_of_squared_error, "RE: ", regularization_error, "TE: ", total_error)

        return total_error

    def fit(self, x, y = None, max_iter = None):
        self.__print_model_params()

        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.__setup_centers()
        self.__setup_weights()

        iteration = 0
        iterate = True
        accuracy = None

        while iterate:
            print("Iteration: ", iteration)

            if (max_iter != None and iteration >= max_iter):
                break

            iteration += 1

            # if minimization_block == 0:
            #     minimize centers and weights
            # else if minimization_block == 1:
            #     minimize centers
            # else if minimization_block == 2:
            #     minimize linear weights

            print('INFO: Minimizing centers.')

            result_1 = optimize.minimize(
                fun=self.__error_function,
                x0=self.centers.flatten(),
                args=1,
                method=self.solver,
                options={
                    'maxiter': 10
                }
            )

            self.__set_centers_and_weights(result_1.x, mode = 1)

            print('INFO: Minimizing weights.')
            self.__setup_interpolation_matrix(self.X)

            result_2 = optimize.minimize(
                fun = self.__error_function,
                x0 = self.weights,
                args = 2,
                method = self.solver,
                options = {
                    'maxfun' : 1
                }
            )

            self.__set_centers_and_weights(result_2.x, mode = 2)

        return self

    def predict(self, x, y=None):
        interpolation_matrix = self.__setup_interpolation_matrix(x)
        predictions = np.dot(interpolation_matrix, self.weights)

        return self.__to_binary(predictions)


##

X = train_x_data[:900, :]
Y = train_y_data[:900]

rbf_nn = RbfNN(noc = 10, solver = 'L-BFGS-B').fit(X, Y, max_iter = 10)

##

Xte = train_x_data[900:, :]
Yte = train_y_data[900:]

print(confusion_matrix(Yte, rbf_nn.predict(Xte)))

##


X = train_x_data[:, :]
Y = train_y_data[:]

param_grid = {
    'noc' : [5, 10, 20, 30],
    'solver' : ['L-BFGS-B', 'BFGS', 'Powell', 'Newton-CG'],
    'sigma' : [2, 3, 4],
    'rho1' : [1e-3, 1e-4, 1e-5],
    'rho2' : [1e-3, 1e-4, 1e-5]
}

gs = GridSearchCV(
    RbfNN(),
    param_grid,
    n_jobs=-1,
    scoring='accuracy'
).fit(X, y=Y)

print("Grid Search completed")

gs_file = open("data/grid_search_" + str(datetime.datetime.now()) + ".pickle", 'wb')
pickle.dump(gs, gs_file)

##