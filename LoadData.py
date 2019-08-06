import os
import numpy as np

################# HARDCODED INPUTS ######################
CallFolder = 'Raw_Data'

#########################################################
# IMPORT, PREPROCESS AND STORE TRAINING DATA AS NUMPY ARRAY
DataTrain = np.genfromtxt(os.path.join(CallFolder, 'Train_Data.csv'), delimiter=';')
y_train = DataTrain[:,6]
X_train = DataTrain[:,1:6]

#########################################################
# # GENERATE FULL DATA
row, col = X_train.shape
X_train_full = np.full((row, 21), 0., dtype=np.float64)
X_train_full[:, 0:5] = X_train
X_train_full[:, 5:10] = X_train ** 2
X_train_full[:, 10:15] = np.sin(X_train)
X_train_full[:, 15:20] = np.cos(X_train)
X_train_full[:, 20:21] = 1.

print(X_train_full)
    
#########################################################
# STORE DATA AS NUMPY ARRAY
print('X_train:   ', X_train_full.shape)
print('y_train:   ', y_train.shape)
np.save('X_train.npy', X_train_full)
np.save('y_train.npy', y_train)
