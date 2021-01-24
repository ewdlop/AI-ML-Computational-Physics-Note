import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression()
X = np.array([[3.04],[3.64],[4.61],[5.57],[6.74],[7.77]])
Y = np.array([0.91,1.01,1.09,1.11,1.20,1.30])
reg.fit(X,Y)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[5]]))