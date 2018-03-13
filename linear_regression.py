# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:19:05 2018

@author: user

Build logistic regression class to find the optimum theta using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import linregress

class Linear_regression():
    
    def __init__(self, X, y, alpha, lamb):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.lamb = lamb
        self.loss_vs_iter = {}
    def gradient_descent(self):
        '''
        
        1. add a bias feature which is always equal to 1.
        2. set all thetas as 0 (number of theta is the number of features + 1 (bias))
        3. compute the gradient of loss function.
        4. update all thetas (simultaneously otherwise it will affect the gradient.)
        5. repeat until a certain number of iteration or until the gradient reach almost 0.
        '''

        #1
        self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis = 1)
        #2 
        thetas = np.zeros((1, self.X.shape[1]))    
        #3
        for _ in range(1000000):
            M = self.X.shape[0] # number of training example
            if _ %5000 == 0:
                loss =  0.5/M * np.sum((np.dot(thetas, self.X.T) - self.y)**2)
                self.loss_vs_iter[_] = loss
            gradient = (1/M) * np.dot((np.dot(thetas, self.X.T) - self.y), self.X)
            #4
            thetas = thetas - self.alpha * (gradient + (self.lamb/M) * thetas)
        return (thetas, self.loss_vs_iter)
        
        
if __name__ == "__main__":
    # create a fairly linear dataset with 1 feature
    X1 = np.array([i for i in range(100)]).reshape(100,1)
    X2 = np.array([i*2 for i in range(100)]).reshape(100,1)
    X = np.concatenate((X1,X2), axis = 1)   
 
    X = X1
    
    noise = 5* np.random.normal(0,1,X1.shape) + 500
    y = (X1 + noise).flatten()
        
    C = Linear_regression(X, y, 0.0001, 0)
    thetas, losses = C.gradient_descent()
    # plot loss vs number of iteration
    plt.figure()
    plt.title("loss vs number of iteration")
    plt.ylabel("loss")
    plt.xlabel("number of iteration")
    plt.grid()
    plt.plot(list(losses.keys()), list(losses.values()), marker = "o")
    #compare my regression line with sklearn and scipy regression line
    plt.figure()
    ax1 = plt.subplot(211)
    ax1.set_title("X1 vs y")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("y").set_rotation(0)
    ax1.scatter(X1.flatten(), y)
    
    clf = LinearRegression()
    clf = clf.fit(X, y)

    m, c = clf.coef_[0], clf.intercept_
    ax1.plot(X1, list(map(lambda x: m*x + c, X1)), label = "sklearn LinearRegression ({0}x + {1})".format(round(m,4), round(c,4)), marker = "x", color = "r", linewidth = 2) 
#    m, c = linregress(X1.flatten(), y)[:2]
#    ax1.scatter(X1, list(map(lambda x: m*x + c, X1)), label = "scipy")
    ax1.plot(X1, list(map(lambda x: thetas[0][1]*x + thetas[0][0], X1)), label = "gradient descent ({0}x + {1})".format(round(thetas[0][1], 4),  round(thetas[0][0],4)), linestyle = "--", color = "orange", linewidth = 3)   
    ax1.legend()
    ax1.grid()
    
    
    
    ax2 = plt.subplot(212)
    ax2.set_title("X2 vs y")
    ax2.set_xlabel("X2")
    ax2.set_ylabel("y").set_rotation(0)
    ax2.scatter(X2.flatten(), y)
    
    clf = LinearRegression()
    clf = clf.fit(X, y)
    m, c = clf.coef_[0], clf.intercept_
    ax2.plot(X2, list(map(lambda x: m*x + c, X2)), label = "sklearn LinearRegression ({0}x + {1})".format(round(m,4), round(c,4)), marker = "x") 
#    m, c = linregress(X2.flatten(), y)[:2]
#    ax2.scatter(X2, list(map(lambda x: m*x + c, X2)), label = "scipy")
    ax2.plot(X2, list(map(lambda x: thetas[0][1]*x + thetas[0][0], X2)), label ="gradient descent ({0}x + {1})".format(round(thetas[0][1], 4), round(thetas[0][0],4)), linestyle = "--")   
    ax2.legend()
    ax2.grid()
    
    #compare result when there is only one feature
    
    