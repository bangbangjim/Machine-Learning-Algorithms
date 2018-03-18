# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:19:05 2018

@author: user

Build linear regression class to find the optimum theta using gradient descent.
TODO:
    1. compare cost function of ridge and my own code
    2. try visualise (3D) stochastic gradient descent (* smaller number of iter needed as it iterates through every sample size) and batch gradient descent
    3. test lamb on higher order polynormial data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

class Linear_regression():

    def __init__(self, X, y, alpha, lamb, num_iter):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.lamb = lamb
        self.loss_vs_iter = {}
        self.num_iter = num_iter
    def gradient_descent(self):
        '''        
        1. add a bias feature which is always equal to 1.
        2. set all thetas as 0 (number of theta is the number of features + 1 (bias))
        3. compute the gradient of loss function (regulisation is applied when lamb is non-zero).
        4. update all thetas (simultaneously otherwise it will affect the gradient.)
        5. repeat until a certain number of iteration or until the gradient reach almost 0.
        '''

        #1
        self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis = 1)
        #2 
        thetas = np.zeros((1, self.X.shape[1]))    
        #3
        for step in range(self.num_iter):
            # number of training example
            M = self.X.shape[0] 
            #compute the loss every 500 step 
            if step %500 == 0:
                loss =  0.5/M * (np.sum((np.dot(thetas, self.X.T) - self.y)**2) + self.lamb * np.sum(thetas**2))
                self.loss_vs_iter[step] = loss
            #3 
            gradient = (1/M) * np.dot((np.dot(thetas, self.X.T) - self.y), self.X)
            #4
            thetas = thetas - self.alpha * (gradient + (self.lamb/M) * np.sum(thetas))
        return (thetas, self.loss_vs_iter)

        
