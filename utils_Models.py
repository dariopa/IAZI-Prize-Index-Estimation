from sklearn.neural_network import MLPRegressor as MLP
from sklearn.linear_model import Ridge as RDG
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso as LS

class Regression():
    def LinRegression(X_data, y_data):
        model =LR(fit_intercept=True,
                normalize=False,
                copy_X=True)
        model.fit(X_data, y_data)

        weights = model.coef_
        return weights

    def RidgeRegression(X_data, y_data):
        model =RDG(alpha=1.0,
                copy_X=True,
                fit_intercept=True,
                max_iter=10000,
                solver='svd',
                tol=0.0001)
        model.fit(X_data, y_data)
        
        weights = model.coef_
        return weights

    def LassoRegression(X_data, y_data):
        model =LS(alpha=1.0,
                max_iter=20000,
                tol=0.0001)
        model.fit(X_data, y_data)

        weights = model.coef_
        return weights

