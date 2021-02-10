# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:38:19 2020

@author: 91877
"""
import loadmodel

from Layer import RNN
import pickle
import numpy as np


def saveweight(model):
    #save the weights 
    params= model
    with open('weightRNN.dat','wb') as f:
        pickle.dump(params,f)
    print('weights saved successfully')
    pass
    
    
    
    
def fit(X,Y,model):
    #train the model
    for epoch in range(model.epoch):
        
        enc_hidden=model.encoder_forward() #encoder hidden state is captured
        y_prob,dec_out,dec_hidden,inp=model.decoder_forward(enc_hidden[model.length-1])  #last encoder hidden state is passed as context vector
        loss=model.loss(y_prob,Y)   #compute loss
        print('loss: ',loss)
        model.decoder_backward(dec_out,dec_hidden,Y,inp)  #backpropagation through time
        model.encoder_backward(enc_hidden)
        model.update_params() #updation of parameters
    saveweight(model)
    pass


def predict(X,Y,embed):
    model=loadmodel.load_weights()  #load the weights
    model.embed_enc=embed
    enc=model.encoder_forward()
    y_pred,dec_out,dec_hidden,dec_inp=model.decoder_forward(enc[-1])
    loss=model.loss(y_pred,Y)
    print(loss)
    return np.argmax(y_pred,axis=0)
    