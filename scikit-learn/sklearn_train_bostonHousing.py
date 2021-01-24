from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
boston = load_boston()
x = boston.data
y = boston.target
# print(x.shape)
# print(y.shape)
# print(x)
# plt.figure(figsize=(4,3))
# plt.hist(y)
# plt.xlabel('price($1000s)')
# plt.ylabel('count')
# plt.tight_layout()
# plt.show()
# for index, feature_name in enumerate(boston.feature_names):
#     plt.figure(figsize=(4,3))
#     plt.scatter(x[:, index], y)
#     plt.ylabel('Price', size=15)
#     plt.xlabel(feature_name, size=15)
#     plt.tight_layout()
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)
# linear = LinearRegression()
# linear.fit(x_train, y_train)
# linear_predicted = linear.predict(x_test)
# plt.figure(figsize =(4,3))
# plt.suptitle('Linear Regression')
# plt.scatter(y_test,linear_predicted)
# plt.plot([0,50],[0,50], '--k')
# plt.axis('tight')
# plt.xlabel('True price($1000s)')
# plt.ylabel('Predicted price($1000s)')
# plt.tight_layout()
# plt.show
# print("Linear RMS: %r " % np.sqrt(np.mean((linear_predicted - y_train)) ** 2))
# print("Linear intercept: ")
# print(linear.intercept_)
# print("Linear cofficent: ")
# print(linear.coef_)

# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(x_train, y_train)
# neigh_predicted = neigh.predict(x_test)
# plt.figure(figsize=(4,3))
# plt.suptitle('KNN')
# plt.scatter(y_test, neigh_predicted)
# plt.plot([0,50],[0,50],'--k')
# plt.axis('tight')
# plt.xlabel('True price ($1000s)')
# plt.ylabel('Predicted price ($1000s)')
# plt.tight_layout()
# plt.show()
# print("KNN RMS: %r " % np.sqrt(np.mean((neigh_predicted - y_test) ** 2)))

# from sklearn import tree
# tree = tree.DecisionTreeRegressor()
# tree.fit(x_train, y_train)
# print('Decision Tree Feature Importance: ')
# print(tree.feature_importances_)
# tree_predicted = tree.predict(x_test)
# plt.figure(figsize=(4, 3))
# plt.suptitle('Decision Tree')
# plt.scatter(y_test, tree_predicted)
# plt.plot([0,50],[0,50], '--k')
# plt.axis('tight')
# plt.xlabel('True price ($1000s)')
# plt.ylabel('Predicted price ($1000s)')
# plt.tight_layout()
# plt.show()
# print("Decision Tree RMS: %r " % np.sqrt(np.mean((tree_predicted - y_test) ** 2)))

# from sklearn.ensemble import RandomForestRegressor
# forest = RandomForestRegressor(max_depth=2, random_state=0)
# forest.fit(x_train, y_train)
# print('Random Forest Feature Importance')
# print(forest.feature_importances_)
# forest_predicted = forest.predict(x_test)
# plt.figure(figsize=(4,3))
# plt.suptitle('Random Forest')
# plt.scatter(y_test, forest_predicted)
# plt.plot([0,50],[0,50],'--k')
# plt.axis('tight')
# plt.xlabel('True price ($1000s)')
# plt.ylabel('Predicted price ($1000s)')
# plt.tight_layout()
# plt.show()
# print("Forest RMS: %r " % np.sqrt(np.mean((forest_predicted - y_test) ** 2)))


from sklearn import datasets
from sklearn.model_selection import cross_val_score
import numpy as np
digits = datasets.load_digits()
x = digits.data
y = digits.target

from sklearn.linear_model import Perceptron
perceptron_model = Perceptron(tol=1e-3, random_state=0)
Perceptron_scores = cross_val_score(perceptron_model, x,y, cv=10)
print('Perceptron avg performance: ')
print(np.mean(perceptron_model))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_scores = cross_val_score(neigh, x,y, cv=10)
print('KNN avg performance: ')
print(np.mean(neigh))

#the same for decsion tree and random forest