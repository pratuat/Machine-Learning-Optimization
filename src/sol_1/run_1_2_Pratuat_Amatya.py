import pickle
import datetime
import numpy as np
import pandas as pd

from src.sol_1.lib.rbf_nn_bc_bd import RbfNNBCBD
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.lib.pushover import notify_me

##

train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]

train_1.iloc[:, 0] = 0 # (1005, 257)
train_8.iloc[:, 0] = 1 # (542, 257)

train_data = np.asanyarray(pd.concat([train_1.iloc[:, :], train_8.iloc[:, :]], axis=0, ignore_index=True))

X = normalize(train_data[:, 1:])
Y = train_data[:, 0]

##

param_grid = {
    'noc' : [4],
    'solver' : ['L-BFGS-B'],
    'sigma' : [2],
    'rho' : [1e-5]
}

gs = GridSearchCV(
    RbfNNBCBD(),
    param_grid,
    n_jobs=-1,
    scoring='accuracy'
).fit(X, y=Y)

print("Grid Search completed")
print(gs.cv_results_)

output_file = "data/output/rbf_nn_bc_bd" + str(datetime.datetime.now()) + ".pickle"
pickle.dump(gs, open(output_file, 'wb'))

##

model = RbfNNBCBD(noc = 4, solver = 'L-BFGS-B', sigma = 2, rho = 1e-5, epsilon=1e-4).fit(X, Y)

##

test_1 = pd.read_csv('data/Test_1.csv').iloc[:, :257]
test_8 = pd.read_csv('data/Test_8.csv').iloc[:, :257]

test_1.iloc[:, 0] = 0 # (1005, 257)
test_8.iloc[:, 0] = 1 # (542, 257)

test_data = np.asanyarray(pd.concat([test_1.iloc[:, :], test_8.iloc[:, :]], axis=0, ignore_index=True))

Xt = normalize(test_data[:, 1:])
Yt = test_data[:, 0]

##

print(confusion_matrix(Yt.astype(int), model.predict(Xt)))

##
