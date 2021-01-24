from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(100):
    ax = fig.add_subplot(10,10,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,7,str(digits.target[i]),color='green')
plt.show()
x= digits.data
y= digits.target
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import Perceptron
perceptron_model = Perceptron(tol=1e-3,random_state=0)
perceptron_model.fit(xtrain,ytrain)
perceptron_prediction=perceptron_model.predict(xtest)
from sklearn import metrics