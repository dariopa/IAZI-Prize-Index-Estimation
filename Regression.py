import os
import numpy as np
from sklearn.utils import shuffle
from utils_Models import Regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils_OutputFormat import PrintOutput
import matplotlib.pyplot as plt


################# HARDCODED INPUTS ######################
CallFolder = 'Raw_Data'

#########################################################
# IMPORT TRAINING DATA
DataTrain = np.genfromtxt(os.path.join(CallFolder, 'Train_Data.csv'), delimiter=';')
y_train = DataTrain[:,6]
X_train = DataTrain[:,1:6]

#########################################################
# PREPROCESS TRAINING DATA
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)

#########################################################
# # GENERATE POLYNOMIAL FUNCTION OF THIRD ORDER
row, col = X_train.shape
X_train_full = np.full((row,16), 0., dtype=np.float32)
X_train_full[:, 0:5] = X_train
X_train_full[:, 5:10] = X_train**2
X_train_full[:, 10:15] = X_train**3
X_train_full[:, 15:16] = 1.

X_train = X_train_full

#########################################################
# TRAIN DATA
weights_lin = Regression.LinRegression(X_train, y_train)
weights_ridge = Regression.RidgeRegression(X_train, y_train)
weights_lasso = Regression.LassoRegression(X_train, y_train)

#########################################################
# DEFINE BIAS
results_lin = X_train.dot(weights_lin)
weights_lin[-1] = np.mean(y_train - results_lin) # calculate bias and add it to end of linear model
results_lin = X_train.dot(weights_lin)
results_lin_comp = np.c_[ results_lin, y_train ] 
print("Linear regression: \n", results_lin_comp)
linear_correlation = np.corrcoef(results_lin_comp[:, 0], results_lin_comp[:, 1])
print("\n Correlation: \n", linear_correlation, "\n")

results_ridge = X_train.dot(weights_ridge)
weights_ridge[-1] = np.mean(y_train - results_ridge) # calculate bias and add it to end of ridge model
results_ridge = X_train.dot(weights_ridge)
results_ridge_comp = np.c_[ results_ridge, y_train ] 
print("\n Ridge regression: \n", results_ridge_comp)
ridge_correlation = np.corrcoef(results_ridge_comp[:, 0], results_ridge_comp[:, 1])
print("\n Correlation: \n", ridge_correlation, "\n")

results_lasso = X_train.dot(weights_lasso)
weights_lasso[-1] = np.mean(y_train - results_lasso) # calculate bias and add it to end of lasso model
results_lasso = X_train.dot(weights_lasso)
results_lasso_comp = np.c_[ results_lasso, y_train ] 
print("\n Lasso regression: \n", results_lasso_comp)
lasso_correlation = np.corrcoef(results_lasso_comp[:, 0], results_lasso_comp[:, 1])
print("\n Correlation: \n", lasso_correlation, "\n")

#########################################################
# STORE WEIGHTS
PrintOutput(weights_lin, os.path.join(CallFolder, "weights_lin.csv"))
# print("Linear weights: ", weights_lin)

PrintOutput(weights_ridge, os.path.join(CallFolder, "weights_ridge.csv"))
# print("Ridge weights: ", weights_ridge)

PrintOutput(weights_lasso, os.path.join(CallFolder, "weights_lasso.csv"))
# print("Lasso weights: ", weights_lasso)

#########################################################
# PLOT RESULTS
"""
plt.figure()
plt.plot(np.arange(1989, 2019, 1), results_lin_comp) 
plt.title('Comparison linear regression with real IAZI Index')
plt.xlabel('Year 1989 - 2018')
plt.ylabel('IAZI Index')
plt.gca().legend(('Estimated IAZI', 'Original IAZI'))
plt.savefig('iazi_index_linear.jpg')

plt.figure()
plt.plot(np.arange(1989, 2019, 1), results_ridge_comp) 
plt.title('Comparison ridge regression with real IAZI Index')
plt.xlabel('Year 1989 - 2018')
plt.ylabel('IAZI Index')
plt.gca().legend(('Estimated IAZI', 'Original IAZI'))
plt.savefig('iazi_index_ridge.jpg')

plt.figure()
plt.plot(np.arange(1989, 2019, 1), results_lasso_comp) 
plt.title('Comparison lasso regression with real IAZI Index')
plt.xlabel('Year 1989 - 2018')
plt.ylabel('IAZI Index')
plt.gca().legend(('Estimated IAZI', 'Original IAZI'))
plt.savefig('iazi_index_lasso.jpg')
"""

# Subplots for better comparison
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(np.arange(1989, 2019, 1), results_lin_comp) 
plt.title('Comparison linear regression with real IAZI Index')
plt.xlabel('Year 1989 - 2018')
plt.ylabel('IAZI Index')
plt.gca().legend(('Estimated IAZI', 'Original IAZI'))

plt.subplot(3,1,2)
plt.plot(np.arange(1989, 2019, 1), results_ridge_comp) 
plt.title('Comparison ridge regression with real IAZI Index')
plt.xlabel('Year 1989 - 2018')
plt.ylabel('IAZI Index')
plt.gca().legend(('Estimated IAZI', 'Original IAZI'))

plt.subplot(3,1,3)
plt.plot(np.arange(1989, 2019, 1), results_lasso_comp) 
plt.title('Comparison lasso regression with real IAZI Index')
plt.xlabel('Year 1989 - 2018')
plt.ylabel('IAZI Index')
plt.gca().legend(('Estimated IAZI', 'Original IAZI'))
plt.tight_layout()
plt.show(1)

