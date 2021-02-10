# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:38:34 2020

@author: 91877
"""


import pickle
#load weights
def load_weights():
    with open('weightRNN.dat','rb') as f:
        weights = pickle.load(f)
    return weights