# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:22:53 2018

@author: Capco
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:19:05 2018

@author: user

Build logistic regression class to find the optimum theta using gradient descent.
TODO:
    1. plot x vs y on the resulting theta 
    2. try skewing the classes
    3. compare to sklearn.
    4. stochastic - done
    5. plot 3d graph of gradient descent of batch and stoch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from scipy.stats import linregress
import logging
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import preprocessing
#from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(filename='logistic_regression.log',level=logging.DEBUG)

class Logistic_regression():
    
    
    def __init__(self, X, y, alpha, lamb, num_iter):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.lamb = lamb
        self.num_iter =num_iter
        self.loss_vs_iter = {}
    def sigmoid(self, z = np.array):
        h = 1/(1+np.exp(-z))
        return h
        
    def SGD(self):
        '''
        1. add a bias feature which is always equal to 1.
        2. randomly shuffle the dataset (X and y)
        3. set all thetas as 0 (number of theta is the number of features + 1 (bias))        
        4. iteratively, compute the gradient for each sample size.
        5. update all thetas after each computation.
        6. repeat until a certain number of iteration or until the gradient reach almost 0.
        '''
        #1, #2
        self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis = 1)
        dataset = np.concatenate((self.X, self.y), axis = 1)
        np.random.shuffle(dataset)
        self.X = np.array([[i[0],i[1]] for i in dataset])
        self.y = np.array([[i[2],] for i in dataset])
        #3
        thetas = np.zeros((1, self.X.shape[1])).flatten()

        #4
        for n in range(self.num_iter):
            M = self.X.shape[0]
            loss =  0
            for i in range(M):
                # compute cost function
                h = self.sigmoid(np.dot(thetas, self.X[i]))  
#                logging.debug(h)
#                loss += 0.5* (h - self.y[i])**2
                loss += -1* np.sum(self.y[i] * np.log(h) + (1-self.y[i])* np.log(1-h)) + (self.lamb/2) * np.sum(thetas **2)
#                logging.debug(loss)
                #perform gradient descent
                gradient = (h - self.y[i]) * self.X[i]
#                logging.debug(gradient)
                thetas = thetas - self.alpha * (gradient + (self.lamb/2) * thetas)
#                logging.debug(thetas)
#                self.loss_vs_iter[(n+1)*M] = (1/M)*loss

            if n % 30 == 0:
                self.loss_vs_iter[(n+1)*M] = (1/M)*np.float(loss)
                
                    
        return (thetas, self.loss_vs_iter)
        
                   
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
        for _ in range(self.num_iter):
            M = self.X.shape[0] # number of training example
            if _ %5000 == 0:
#            if type(_) == int:
                #compute loss function
                h = self.sigmoid(np.dot(thetas, self.X.T))                 
                loss = (-1/M) * np.sum(self.y.T * np.log(h) + (1-self.y.T) * np.log(1-h)) + (self.lamb/(2*M)) * np.sum(thetas**2)
                self.loss_vs_iter[_] = loss
            #3, #4 perform gradient descent
            gradient = (1/M) * np.dot((self.sigmoid(np.dot(thetas, self.X.T)) - self.y.T), self.X)
            thetas = thetas - self.alpha * (gradient + (self.lamb/M) * thetas)
        return (thetas, self.loss_vs_iter)
        
        
if __name__ == "__main__":
    # create a dataset where if x <50 y = 0, if x > 60 y = 1
    X1 = np.array([[i,] for i in range(100) if i < 50 or i >= 60])
    y = np.array([[0,] if i <50 else [1,] for i in X1])
    

        
    C = Logistic_regression(X1, y, 0.01, 0, 1000000)
    thetas, losses = C.gradient_descent()
    print ("C done")
    D = Logistic_regression(X1, y, 0.003, 0, 20000)
    t, l = D.SGD()
    
    clf = LogisticRegression()
    clf = clf.fit(X1, y.flatten())
    m, c = map(np.float, [clf.coef_, clf.intercept_])
    
    
    
    # plot loss vs number of iteration for batch and stochastic
    plt.figure()
    plt.title("loss vs number of iteration")
    plt.ylabel("loss")
    plt.xlabel("number of iteration")
    plt.grid()
    plt.plot(list(losses.keys()), list(losses.values()), marker = "o", label = "batch")
    plt.plot(list(l.keys()), list(l.values()), marker = "x", label = "stochastic")
    plt.legend()
    # plot X vs y of the result model
    plt.figure()
    x = np.array([[1,i] for i in range(100)]).reshape(100,2)
    plt.scatter(X1, y)
    plt.plot([i[1] for i in x], C.sigmoid(np.dot(thetas, x.T)).flatten(), label = "batch thetas: " + (str([round(x,2) for x in thetas.flatten()])))
    plt.plot([i[1] for i in x], D.sigmoid(np.dot(t, x.T)).flatten(), label = "stochastic thetas: " + (str([round(x,2) for x in t.flatten()])))
    x_wihout_bias = np.array([i[1] for i in x]).reshape(-1,1)
    plt.plot(x_wihout_bias, clf.predict(x_wihout_bias), label = "sklearn LogisticRegression thetas: "+ ", ".join(map(lambda x: str(round(x,2)), [m,c])))
    
    plt.legend()
    plt.grid()
    plt.title("Logistic Regression")
    plt.ylabel("y").set_rotation(0)
    plt.xlabel("X1")
    
    #create a dataset with 2 features where a decision boundary can be drawn
    
    X1 = np.array([[i,] for i in range(100) if i < 40 or i >= 60])
    noise = np.random.normal(0,1, X1.shape) * 10 
    X2 = X1.copy() + noise
    X = np.concatenate((X1,X2), axis = 1)
    y = np.array([[0,] if np.product(i) <2500 else [1,] for i in X])
    
    #plot the data to confirm there is a decision boundary
    plt.figure()
    plt.scatter(np.array([X1[i] for i in range(len(X1)) if y[i] == 0]), np.array([X2[i] for i in range(len(X2)) if y[i] == 0]))
    plt.scatter(np.array([X1[i] for i in range(len(X1)) if y[i] == 1]), np.array([X2[i] for i in range(len(X2)) if y[i] == 1]))
    
         
    ax1.scatter(X1.flatten(), X2.flatten(), y.flatten())
    ax1.set_ylabel = "X2"
    ax1.set_xlabel = "X1"
    ax1.set_zlabel = "label"
    
    clf = LogisticRegression()
    clf = clf.fit(X, y.flatten())
    
    X_test = np.array([[i,i] for i in range(100)])
    y_test = clf.predict(X_test)
    
    # plot the graph of sklearn , my own algo to a 3d graph
    fig = plt.figure()
    ax1 = fig.gca(projection = "3d")    
    ax1.plot([i for i in range(100)], [i for i in range(100)], y_test)    
    
    #todo - build cost function 3d graph to demonstrate gradient descent
    fig = plt.figure()
    ax1 = fig.gca(projection = "3d")    
    
    
# =============================================================================
#     dataset = load_breast_cancer()
#     df = pd.DataFrame(data = dataset["data"], columns = load_breast_cancer()["feature_names"])
#     y = pd.DataFrame(data = dataset["target"], columns = [["label"]]).values
#     # only use the first two features for simplication purpose
#     X = df[df.columns[:2]]
#     
# =============================================================================
    #plot the 3D graph to confirm that the data 
    
    
    
    
    
# =============================================================================
#     #compare my regression line with sklearn and scipy regression line
#     plt.figure()
#     ax1 = plt.subplot(211)
#     ax1.set_title("X1 vs y")
#     ax1.set_xlabel("X1")
#     ax1.set_ylabel("y").set_rotation(0)
#     ax1.scatter(X1.flatten(), y)
#     
#     clf = LinearRegression()
#     clf = clf.fit(X, y)
# 
#     m, c = clf.coef_[0], clf.intercept_
#     ax1.plot(X1, list(map(lambda x: m*x + c, X1)), label = "sklearn LinearRegression ({0}x + {1})".format(round(m,4), round(c,4)), marker = "x", color = "r", linewidth = 2) 
# #    m, c = linregress(X1.flatten(), y)[:2]
# #    ax1.scatter(X1, list(map(lambda x: m*x + c, X1)), label = "scipy")
#     ax1.plot(X1, list(map(lambda x: thetas[0][1]*x + thetas[0][0], X1)), label = "gradient descent ({0}x + {1})".format(round(thetas[0][1], 4),  round(thetas[0][0],4)), linestyle = "--", color = "orange", linewidth = 3)   
#     ax1.legend()
#     ax1.grid()
#     
#     
#     
#     ax2 = plt.subplot(212)
#     ax2.set_title("X2 vs y")
#     ax2.set_xlabel("X2")
#     ax2.set_ylabel("y").set_rotation(0)
#     ax2.scatter(X2.flatten(), y)
#     
#     clf = LinearRegression()
#     clf = clf.fit(X, y)
#     m, c = clf.coef_[1], clf.intercept_
#     ax2.plot(X2, list(map(lambda x: m*x + c, X2)), label = "sklearn LinearRegression ({0}x + {1})".format(round(m,4), round(c,4)), marker = "x") 
# #    m, c = linregress(X2.flatten(), y)[:2]
# #    ax2.scatter(X2, list(map(lambda x: m*x + c, X2)), label = "scipy")
#     ax2.plot(X2, list(map(lambda x: thetas[0][2]*x + thetas[0][0], X2)), label ="gradient descent ({0}x + {1})".format(round(thetas[0][2], 4), round(thetas[0][0],4)), linestyle = "--")   
#     ax2.legend()
#     ax2.grid()
#     
#     #compare result when there is only one feature
#     
#     
# =============================================================================
