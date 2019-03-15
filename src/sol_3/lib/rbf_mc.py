import numpy as np
from sklearn.base import BaseEstimator
from src.sol_1.lib.rbf import Rbf

class RbfMC(BaseEstimator):
    def __init__(self, noc = 4, solver = 'L-BFGS-B', sigma = 2, rho = 1e-5):
        self.noc = noc
        self.solver = solver
        self.sigma = sigma
        self.rho = rho

        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.models = None

    def fit(self, x, y=None):
        self.X = x
        self.Y = y
        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]

        self.models = []

        for i in range(self.Y_dim):
            model = Rbf(
                noc = self.noc,
                solver = self.solver,
                sigma = self.sigma,
                rho = self.rho
            )

            model.fit(x = self.X, y = self.Y[:, i])
            self.models.append(model)

        return self

    def predict(self, x, y=None):
        prediction = []

        for model in self.models:
            prediction.append(model.predict(x))

        return np.array(prediction).T
