#%%
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from src.sol_3.lib.rbf_nn_mc import RbfNNMC

#%%

train_1 = pd.read_csv('data/Train_1.csv').iloc[:, :257]
train_2 = pd.read_csv('data/Train_2.csv').iloc[:, :257]
train_8 = pd.read_csv('data/Train_8.csv').iloc[:, :257]

train_data = np.asanyarray(
    pd.concat(
        [
            train_1.iloc[:, :],
            train_2.iloc[:, :],
            train_8.iloc[:, :]
        ], axis=0, ignore_index=True
    )
)

train_x_data = normalize(train_data[:, 1:])
train_y_data = train_data[:, 0]

#%%

label_binarizer = LabelBinarizer()

X = train_x_data[:, :]
Y = label_binarizer.fit_transform(train_y_data[:])

model = RbfNNMC(5).fit(X, Y)
preds = model.predict(X)

confusion_matrix(train_y_data[:], label_binarizer.inverse_transform(preds))

#%%

gs_file = open("data/grid_search_" + str(datetime.datetime.now()) + ".pickle", 'wb')
pickle.dump(gs, gs_file)

#%%
