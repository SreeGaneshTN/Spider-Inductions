# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 00:12:45 2020

@author: 91877
"""

import numpy as np

class Layer:
    
    def __init__(self,no_layers,num_nodes:list,activation:list):
        self.layers=no_layers
        self.nodes=num_nodes
        self.activation=activation
        self.params=self.intialize_weight()
        self.grads={}
    def intialize_weight(self):
        np.random.seed(3)
        parameters = {}
        L = len(self.nodes)       

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.nodes[l],self.nodes[l-1])*0.01
            parameters['b' + str(l)] = np.zeros((self.nodes[l],1))
            
            
        assert(parameters['W' + str(l)].shape == (self.nodes[l], self.nodes[l-1]))
        assert(parameters['b' + str(l)].shape == (self.nodes[l], 1))

        
        return parameters

