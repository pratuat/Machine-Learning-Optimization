import pickle
import datetime
import numpy as np
import pandas as pd

from src.sol_1.lib.rbf import Rbf
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

##

# Loading Training Data
train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]

train_1.iloc[:, 0] = 0
train_8.iloc[:, 0] = 1

train_data = np.asanyarray(pd.concat([train_1.iloc[:, :], train_8.iloc[:, :]], axis=0, ignore_index=True))

X = normalize(train_data[:, 1:])
Y = train_data[:, 0]

##

param_grid = {
    'noc' : [2],
    'solver' : ['L-BFGS-B', 'BFGS', 'TNC', 'CG', 'Nelder-Mead'],
    'sigma' : [2, 3, 4],
    'rho' : [1e-4, 1e-5, 1e-6]
}

gs = GridSearchCV(
    Rbf(),
    param_grid,
    n_jobs=-1,
    scoring='accuracy'
).fit(X, Y)

##
print("!!! Grid Search completed !!!")
print()
print("Best Model Paramters:")
print('\tNumber of neurons N:', gs.best_params_['noc'])
print('\tSolver:', gs.best_params_['solver'])
print('\tValue of σ:', gs.best_params_['sigma'])
print('\tValue of ρ:', gs.best_params_['rho'])
print('\tAccuracy:', gs.best_score_)
##

