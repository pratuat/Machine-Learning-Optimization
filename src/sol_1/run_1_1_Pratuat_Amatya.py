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

# Loading Test Data
test_1 = pd.read_csv('data/Test_1.csv').iloc[:, :257]
test_8 = pd.read_csv('data/Test_8.csv').iloc[:, :257]

test_1.iloc[:, 0] = 0 # (1005, 257)
test_8.iloc[:, 0] = 1 # (542, 257)

test_data = np.asanyarray(pd.concat([test_1.iloc[:, :], test_8.iloc[:, :]], axis=0, ignore_index=True))

Xt = normalize(test_data[:, 1:])
Yt = test_data[:, 0]

##

param_grid = {
    'noc' : [4],
    'solver' : ['L-BFGS-B'],
    'sigma' : [2],
    'rho' : [1e-5]
}

gs = GridSearchCV(
    Rbf(),
    param_grid,
    n_jobs=-1,
    scoring='accuracy'
).fit(X, Y)

print("Grid Search completed")
# print(gs.cv_results_)

# output_file = "data/output/rbf_nn_bc_" + str(datetime.datetime.now()) + ".pickle"
# pickle.dump(gs, open(output_file, 'wb'))
#
# # notify_me("|| Gridsearch Completed ||", 1)

##

model = Rbf(noc = 4, solver = 'L-BFGS-B', sigma = 2, rho = 1e-5).fit(X, Y)
predictions = model.test(Xt, Yt)

##


_ = [print(*k) for k in model.statistics()]
##

