import pandas as pd
import numpy as np
x_train_tensor = np.load("Data/xtrain_spec.npy")
y_train = np.load("Data/ytrain_spec.npy")
tensor_shape = x_train_tensor.shape
x_train = x_train_tensor.flatten().reshape(tensor_shape[0], tensor_shape[1]*tensor_shape[2])
print("Shape: ",x_train.shape)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
print(rf.score(x_train, y_train))