import pickle
import datetime
import numpy as np
import pandas as pd

from src.sol_1.lib.rbf_nn_bc import RbfNNBC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV

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
    'noc' : [5, 10, 20, 30],
    'solver' : ['L-BFGS-B', 'BFGS', 'Powell', 'Newton-CG'],
    'sigma' : [2, 3, 4],
    'rho' : [1e-3, 1e-4, 1e-5]
}

gs = GridSearchCV(
    RbfNNBC(),
    param_grid,
    n_jobs=-1,
    scoring='accuracy'
).fit(X, y=Y)

print("Grid Search completed")
print(gs.cv_results_)

output_file = "data/output/rbf_nn_bc_" + str(datetime.datetime.now()) + ".pickle"
pickle.dump(gs, open(output_file, 'wb'))

##

