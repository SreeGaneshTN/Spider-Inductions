# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:10:28 2020

@author: 91877
"""
import pickle

def load_weights():
    with open('weights.dat','rb') as f:
        weights = pickle.load(f)
        net1,net2=weights[0],weights[1]
    return net1,net2
        