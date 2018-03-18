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

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from scipy.stats import linregress
import logging
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='logistic_regression.log',level=logging.DEBUG)

class Logistic_regression():
        
    def __init__(self, X, y, alpha, lamb, num_iter):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.lamb = lamb
        self.num_iter =num_iter
        self.loss_vs_iter = {}
        self.thetas = None
        
        
    def sigmoid(self, z = np.array):
        h = 1/(1+np.exp(-z))
        return h
        
    def predict(self, x_test):
        x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis = 1)        
        y_predict = np.round(self.sigmoid(np.dot(self.thetas, x_test.T)).flatten())
        
        
        return y_predict
    def score(self, X, y):
        y_predict = self.predict(X)
        accuracy = np.sum((y_predict.flatten() == y.flatten()))/len(y_predict)        
        return accuracy
        
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
        dataset = np.concatenate((self.X, self.y.reshape(-1,1)), axis = 1)
        np.random.shuffle(dataset)
        self.X, self.y = np.hsplit(dataset, [3,])
#        self.X = np.hsplit(dataset, -1)
#        self.X = np.array([[i[0],i[1]] for i in dataset])
#        self.y = np.array([[i[2],] for i in dataset])
        #3
        self.thetas = np.zeros((1, self.X.shape[1])).flatten()

        Xs = [self.thetas,]
        Zs = []
        #4
        for n in range(self.num_iter):
            M = self.X.shape[0]
            loss =  0
            for i in range(M):
                # compute cost function
                z = np.dot(self.thetas, self.X[i])
                h = self.sigmoid(z) 
                if h == 0:
                    h = 0.0000000001
                elif h == 1:
                    h = 0.9999999999
                else:
                    pass              
                output_difference = h - self.y[i]

                gradient =  output_difference * self.X[i]
                self.thetas = self.thetas - self.alpha * (gradient + (self.lamb/2) * np.sum(self.thetas))
                loss += -1* np.sum(self.y[i] * np.log(h) + (1-self.y[i])* np.log(1-h)) + (self.lamb/2) * np.sum(self.thetas **2)
                Xs.append(self.thetas)
                Zs.append(loss)              

#            if n % 30 == 0:
            self.loss_vs_iter[(n+1)*M] = (1/M)*np.float(loss)               
                    
        return (self.thetas, self.loss_vs_iter, np.concatenate(Xs).reshape(-1,3), Zs)
                   
    def gradient_descent(self):
        '''
        
        1. add a bias feature which is always equal to 1.
        2. set all thetas as 0 (number of theta is the number of features + 1 (bias))
        3. compute the gradient of loss function.
        4. update all thetas (simultaneously otherwise it will affect the gradient.)
        5. repeat until a certain number of iteration or until the gradient reach almost 0.
        
        * note try removing the M to increase the speed
        '''

        #1
        self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis = 1)
        #2 
        self.thetas = np.zeros((1, self.X.shape[1]))    
        Xs = [self.thetas,]
        Zs = []
        M = self.X.shape[0] # number of training example
        for step in range(self.num_iter):
#            print (step)
            
            #3, #4 perform gradient descent
            z = np.dot(self.thetas, self.X.T)            
            h = self.sigmoid(z) 
            # this is to avoid log(0) from happening which will give nan when multiplying by 0 later on and cant calculate the loss
            h[h == 1] = 0.9999999999
            h[h == 0] = 0.0000000001
            output_difference = h - self.y.T
#            gradient = (1/M) * np.dot(output_difference, self.X)
            gradient = np.dot(output_difference, self.X)
#            self.thetas = self.thetas - self.alpha * (gradient + (self.lamb/M) * self.thetas)
            self.thetas = self.thetas - self.alpha * (gradient + (self.lamb) * np.sum(self.thetas))
            
            if step %100 == 0:
#            if step < 10000:
#                print (step)
                #compute loss function
                
                loss = (-1/M) * np.sum(self.y.T * np.log(h) + (1-self.y.T) * np.log(1-h)) + (self.lamb/(2*M)) * np.sum(self.thetas**2)
                
                Xs.append(self.thetas)
                Zs.append(loss)
                
#                loss = -1 * np.sum(self.y.T * np.log(h) + (1-self.y.T) * np.log(1-h)) + (self.lamb/(2*M)) * np.sum(self.thetas**2)
                self.loss_vs_iter[step] = loss            
        return (self.thetas, self.loss_vs_iter, np.concatenate(Xs), Zs)
    def cost_function(self, x, y):
         '''
         create a range of thetas and use it to compute the cost function (batch) with the data X, y 
         '''
         
#         self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis = 1)
         # create a range of thetas

         thetas_set = np.array([[self.thetas[0][0],i,j] for i in x for j in y])
         M = self.X.shape[0]
         h = self.sigmoid(np.dot(thetas_set, self.X.T))   
         h[h == 1] = 0.9999999999
         h[h == 0] = 0.0000000001                
         loss = (-1/M) * np.sum(self.y.T * np.log(h) + (1-self.y.T) * np.log(1-h),axis = 1) + (self.lamb/(2*M)) * np.sum(thetas_set**2)

         return (x, y, loss.reshape(x.size, y.size)) 
    def decision_boundary(self, x = np.array, y = np.array):
        '''return z values for plotting decision boundary'''
        if type(self.thetas) == None:
            self.gradient_descent()
        else:
            pass
        Xs = np.array([[i,j] for i in x for j in y])
        z = self.predict(Xs).reshape(x.size, y.size)
        return z
         