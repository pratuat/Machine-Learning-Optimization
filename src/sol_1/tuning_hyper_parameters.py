import pickle
import datetime
import numpy as np
import pandas as pd

from src.sol_2.lib.svm import SVM
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
test_1 = pd.read_csv('data/Test_1.csv').iloc[:, :257]
test_8 = pd.read_csv('data/Test_8.csv').iloc[:, :257]

test_1.iloc[:, 0] = 0 # (1005, 257)
test_8.iloc[:, 0] = 1 # (542, 257)

test_data = np.asanyarray(pd.concat([test_1.iloc[:, :], test_8.iloc[:, :]], axis=0, ignore_index=True))

Xt = normalize(test_data[:, 1:])
Yt = test_data[:, 0]
##

param_grid = {
    'noc' : list(range(2, 5, 1)),
    'solver' : ['L-BFGS-B', 'BFGS', 'TNC', 'CG', 'Nelder-Mead'],
    'sigma' : list(np.arange(1, 3.1, 0.5)),
    'rho' : [1e-6, 1e-7, 1e-8]
}

gs = GridSearchCV(
    SVM(),
    param_grid,
    n_jobs=-1,
    scoring='accuracy',
    error_score=1
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
