import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import pickle
from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify


iris=load_iris()
iris.data = np.delete(iris.data, [1,3], 1)
iris.feature_names = iris.feature_names[::2]
iris.data = np.delete(iris.data, np.where(iris.target==2), axis=0)
iris.target = np.delete(iris.target, np.where(iris.target==2), axis=0)

df = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])
df.head()

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        # zainicjowanie weights
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        # weighted sum
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

model = Perceptron()
X=df.iloc[:,:2].values
y=df.target.values
model.fit(X,y)

with open("model_rta.pkl", "wb") as fh:
    pickle.dump(model, fh)


