import os
import numpy as np
from utils_OutputFormat import PrintOutput

################# HARDCODED INPUTS ######################
CallFolder = 'Raw_Data'
ResultFolder = 'Results_Cases'

#########################################################
# IMPORT WEIGHTS AND CASE STUDIES
weights_lasso = np.genfromtxt(os.path.join(CallFolder, 'weights_lasso.csv'), delimiter=';')
weights_ridge = np.genfromtxt(os.path.join(CallFolder, 'weights_ridge.csv'), delimiter=';')
X_scenarios = np.genfromtxt(os.path.join(CallFolder, 'scenarios.csv'), delimiter=';')
print(X_scenarios.shape)


#########################################################
# GENERATE POLYNOMIAL FUNCTION OF THIRD ORDER
row, col = X_scenarios.shape
X_scenarios_full = np.full((row,16), 0., dtype=np.float32)
X_scenarios_full[:, 0:5] = X_scenarios
X_scenarios_full[:, 5:10] = X_scenarios**2
X_scenarios_full[:, 10:15] = X_scenarios**3
X_scenarios_full[:, 15:16] = 1.

X_scenarios = X_scenarios_full
#########################################################
# CALCULATE IAZI FOR ALL SCENARIOS
lasso_iazi_index = X_scenarios.dot(weights_lasso)
print("IAZI Index for scenarios, calculated with lasso regression: \n", lasso_iazi_index, "\n")
PrintOutput(lasso_iazi_index, os.path.join(ResultFolder, "lasso_iazi_index.csv"))

ridge_iazi_index = X_scenarios.dot(weights_ridge)
print("IAZI Index for scenarios, calculated with ridge regression: \n", ridge_iazi_index, "\n")
PrintOutput(ridge_iazi_index, os.path.join(ResultFolder, "ridge_iazi_index.csv"))


