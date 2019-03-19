import numpy as np
import pandas as pd

from src.sol_1.lib.rbf import Rbf
from sklearn.preprocessing import normalize

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

model = Rbf(noc=4, rho=1e-08, sigma=2, solver='L-BFGS-B').fit(X, Y)

##

predictions = model.test(Xt, Yt)
_ = [print(*k) for k in model.statistics()]

