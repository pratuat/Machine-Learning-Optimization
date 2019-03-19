import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from src.sol_2.lib.svm import SVM

##

# Train Data
train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]

train_1.iloc[:, 0] = 1 # (1005, 257)
train_8.iloc[:, 0] = -1 # (542, 257)

train_data = np.asanyarray(pd.concat([train_1.iloc[:, :], train_8.iloc[:, :]], axis=0, ignore_index=True))

X = normalize(train_data[:, 1:])
Y = train_data[:, 0]

# Test Data
test_1 = pd.read_csv('data/Test_1.csv').iloc[:, :257]
test_8 = pd.read_csv('data/Test_8.csv').iloc[:, :257]
test_1.iloc[:, 0] = 1 # (1005, 257)
test_8.iloc[:, 0] = -1 # (542, 257)

test_data = np.asanyarray(pd.concat([test_1.iloc[:, :], test_8.iloc[:, :]], axis=0, ignore_index=True))

Xt = normalize(test_data[:, 1:])
Yt = test_data[:, 0]

##

model = SVM(gamma=0.02).fit(X, Y)

##

train_results = confusion_matrix(Y.astype(int), model.predict(X))
test_results = confusion_matrix(Yt.astype(int), model.predict(Xt))

_ = [print(*k) for k in model.statistics()]
print("Misclassification rate on training set:", 1 - train_results.diagonal().sum()/train_results.sum())
print("Misclassification rate on test set:", 1 - test_results.diagonal().sum()/test_results.sum())
