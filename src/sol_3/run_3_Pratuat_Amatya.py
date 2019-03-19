import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from src.sol_3.lib.rbf_mc import RbfMC

##

# Loading Training Data
train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_2 = pd.read_csv('data/Train_2.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]

df = pd.concat(
    [
        train_1.iloc[:, :],
        train_2.iloc[:, :],
        train_8.iloc[:, :]
    ], axis=0, ignore_index=True
)
train_data = np.asarray(df, dtype='float')

train_x_data = normalize(train_data[:, 1:])
train_y_data = train_data[:, 0]

## Loading Test Data
test_1 = pd.read_csv('data/Test_1.csv').iloc[:, :257]
test_2 = pd.read_csv('data/Test_2.csv').iloc[:, :257]
test_8 = pd.read_csv('data/Test_8.csv').iloc[:, :257]

df = pd.concat(
    [
        test_1.iloc[:, :],
        test_2.iloc[:, :],
        test_8.iloc[:, :]
    ], axis=0, ignore_index=True
)
test_data = np.asarray(df, dtype='float')

test_x_data = normalize(test_data[:, 1:])
test_y_data = test_data[:, 0]

##

label_binarizer = LabelBinarizer()

X = train_x_data[:, :]
Y = label_binarizer.fit_transform(train_y_data[:])

Xt = test_x_data[:, :]
Yt = label_binarizer.transform(test_y_data[:])

model = RbfMC(noc = 4, solver = 'L-BFGS-B', sigma = 2, rho = 1e-7).fit(X, Y)
train_predictions = model.predict(X)
test_predictions = model.predict(Xt)

##

train_results = confusion_matrix(train_y_data[:], label_binarizer.inverse_transform(train_predictions))
test_results = confusion_matrix(test_y_data[:], label_binarizer.inverse_transform(test_predictions))

##

print("Algorithm used:", 'one-against-all')
print("Optimization solver:", model.solver)
print("Misclassification rate on training set:", 1 - train_results.diagonal().sum()/train_results.sum())
print("Misclassification rate on test set:", 1 - test_results.diagonal().sum()/test_results.sum())
print("Optimization time: ", sum([m.optimization_time for m in model.models]))

##

