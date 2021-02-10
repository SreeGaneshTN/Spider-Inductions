# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:43:37 2020

@author: 91877
"""
import numpy as np

class Activation:
    def Sigmoid(x,derivative=False):
        if derivative:
            return x-(x**2)
        else:
            return 1/(1+np.exp(-x))
    def Tanh(x,derivative=False):
        if derivative:
            return 1-(np.tanh(x)*np.tanh(x))
        else:
            return np.tanh(x)
    def Relu(x,derivative=False):
        if derivative:
            return np.greater(x,0).astype(float)
        else:
            return np.maximum(x,0)
    def Linear(x,derivative=False):
        if derivative:
            return 1
        else:
            return x
    

