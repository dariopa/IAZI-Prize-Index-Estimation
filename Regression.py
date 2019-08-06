import os
import numpy as np
from sklearn.utils import shuffle
from utils_Models import Regression
from utils_OutputFormat import PrintOutput

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
X_train_full = np.full((row,21), 0., dtype=np.float32)
X_train_full[:, 0:5] = X_train
X_train_full[:, 5:10] = X_train**2
X_train_full[:, 10:15] = np.sin(X_train)
X_train_full[:, 15:20] = np.cos(X_train)
X_train_full[:, 20:21] = 1.

X_train = X_train_full
#########################################################
# STORE DATA AS NUMPY ARRAY
np.save(os.path.join(CallFolder, 'X_train.npy'), X_train)
np.save(os.path.join(CallFolder, 'y_train.npy'), y_train)

#########################################################
# LOAD DATA FROM NUMPY ARRAY

X_train = np.load('Raw_Data/X_train.npy')
y_train = np.load('Raw_Data/y_train.npy')

#########################################################
# TRAIN DATA

weights_lin = Regression.LinRegression(X_train, y_train)
weights_ridge = Regression.RidgeRegression(X_train, y_train)
weights_lasso = Regression.LassoRegression(X_train, y_train)

#########################################################
# STORE WEIGHTS

PrintOutput(weights_lin, os.path.join(CallFolder, "weights_lin.csv"))
# print("Linear weights: ", weights_lin)

PrintOutput(weights_ridge, os.path.join(CallFolder, "weights_ridge.csv"))
# print("Ridge weights: ", weights_ridge)

PrintOutput(weights_lasso, os.path.join(CallFolder, "weights_lasso.csv"))
# print("Lasso weights: ", weights_lasso)

#########################################################
# DEFINE BIAS
results_lin = X_train.dot(weights_lin)
weights_lin[-1] = np.mean(y_train - results_lin) # calculate bias and add it to end of linear model
results_lin = X_train.dot(weights_lin)
results_lin_comp = np.c_[ results_lin, y_train ] 
print("Linear regression: \n", results_lin_comp)

results_ridge = X_train.dot(weights_ridge)
weights_ridge[-1] = np.mean(y_train - results_ridge) # calculate bias and add it to end of ridge model
results_ridge = X_train.dot(weights_ridge)
results_ridge_comp = np.c_[ results_ridge, y_train ] 
print("\n Ridge regression: \n", results_ridge_comp)

results_lasso = X_train.dot(weights_lasso)
weights_lasso[-1] = np.mean(y_train - results_lasso) # calculate bias and add it to end of lasso model
results_lasso = X_train.dot(weights_lasso)
results_lasso_comp = np.c_[ results_lasso, y_train ] 
print("\n Lasso regression: \n", results_lasso_comp)


