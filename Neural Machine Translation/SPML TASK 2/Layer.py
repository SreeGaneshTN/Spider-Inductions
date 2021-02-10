# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 23:50:40 2020

@author: 91877
"""
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class RNN:

    def __init__(self,embed_enc,eng_vocab,ger_vocab,epoch,lr,hid_dim,embed_size,timesteps):
        self.embed_enc=embed_enc  #embedded input
        self.epoch=epoch  #number of epoch
        self.lr=lr   #learning rate

        self.embed_size=embed_size   
        self.ger_vocab=ger_vocab
        self.eng_vocab=eng_vocab
        self.length=timesteps
        self.hidden=hid_dim
        
        #Parameter Initialization
        self.wax_enc=np.random.randn(hid_dim,embed_size)*0.01
        self.waa_enc=np.random.randn(hid_dim,hid_dim)*0.01
        self.wba_enc=np.zeros((hid_dim,1))
        self.wax_dec=np.random.randn(hid_dim,1)*0.01
        self.waa_dec=np.random.randn(hid_dim,hid_dim)*0.01
        self.way_dec=np.random.randn(ger_vocab,hid_dim)*0.01
        self.wba_dec=np.zeros((hid_dim,1))
        self.wby_dec=np.zeros((ger_vocab,1))
        self.dwax_enc=np.zeros_like(self.wax_enc)
        self.dwaa_enc=np.zeros_like(self.waa_enc)
        self.dwba_enc=np.zeros_like(self.wba_enc)
        self.dwax_dec=np.zeros_like(self.wax_dec)
        self.dwaa_dec=np.zeros_like(self.waa_dec)
        self.dway_dec=np.zeros_like(self.way_dec)
        self.dwba_dec=np.zeros_like(self.wba_dec)
        self.dwby_dec=np.zeros_like(self.wby_dec)
        self.daa_dec=np.zeros((hid_dim,embed_enc.shape[1]))
        self.daa_enc=np.zeros_like(self.daa_dec)
    def encoder_forward(self):
        #forward prop in encoder
        enc_hidden={}
        initial=np.zeros((self.waa_enc.shape[0],self.embed_enc.shape[1]))
        a_next=initial
        enc_hidden[-1]=a_next
        for t in range(self.length):
            xt=self.embed_enc[:,:,t]
            a_next = np.tanh(np.dot(self.wax_enc,xt)+np.dot(self.waa_enc,a_next)+self.wba_enc)
            enc_hidden[t]=a_next
        return enc_hidden
    
    def decoder_forward(self,enc_hidden):
        #forward prop in decoder
        dec_output=[]
        dec_input=[]
        dec_hidden={}
        dec_hidden[-1]=enc_hidden
        inp=np.zeros(shape=(1,self.embed_enc.shape[1]),dtype=int)
        inp+=self.ger_vocab+1
        # input as start token index
        hidden_next=enc_hidden
        y_pred=np.zeros((self.ger_vocab,self.embed_enc.shape[1],self.embed_enc.shape[2]))
        for t in range(self.length):
            xt=inp 
            hidden_next = np.tanh(np.dot(self.wax_dec,xt)+np.dot(self.waa_dec,hidden_next)+self.wba_dec)
            #compute softmax
            ypred=softmax(np.dot(self.way_dec,hidden_next)+self.wby_dec)
            dec_hidden[t]=hidden_next
            dec_output.append(ypred)
            y_pred[:,:,t]=ypred
            #take the previous output and give it as input by sampling 
            for m in range(self.embed_enc.shape[1]):
                np.random.seed(m)
                inp[:,m]=np.random.choice(list(range(self.ger_vocab)),p=ypred[:,m].ravel())
            dec_input.append(inp)
        return y_pred,dec_output,dec_hidden,dec_input
    
    def loss(self,pred,orig):
        #compute loss
        loss=0
        for t in range(orig.shape[1]):
            for m in range(orig.shape[0]):
                temp=orig[m,t]
                if temp==0:
                    continue
                else:
                    loss+=-np.log(pred[temp,m,t])
        return loss/orig.shape[0]
    
    def decoder_backward(self,out,hidden,orig,dec_input):
        #Backpropagation through time
        for t in reversed(range(self.length)):
            xt=dec_input[t]
            dy=np.copy(out[t])
            Y=orig[:,t]
            for m in range(Y.shape[0]):
                dy[Y[m],m]-=1
            self.dway_dec += np.dot(dy, hidden[t].T)
            self.dwby_dec += np.sum(dy,axis=1,keepdims=True)
            da = np.dot(self.way_dec.T, dy) + self.daa_dec # backprop into h
            daraw = (1 - (hidden[t]**2)) * da # backprop through tanh nonlinearity
            self.dwba_dec+= np.sum(daraw,axis=1,keepdims=True)
            self.dwax_dec += np.dot(daraw, xt.T)
            self.dwaa_dec += np.dot(daraw, hidden[t-1].T)
            self.daa_dec = np.dot(self.waa_dec.T, daraw)
        pass
    
    def encoder_backward(self,hidden):
        self.daa_enc=self.daa_dec
        for t in reversed(range(self.length)):
            xt=self.embed_enc[:,:,t]
            da = self.daa_enc # backprop into h
            daraw = (1 - (hidden[t]**2)) * da # backprop through tanh nonlinearity
            self.dwba_enc+= np.sum(daraw,axis=1,keepdims=True)
            self.dwax_enc += np.dot(daraw, xt.T)
            self.dwaa_enc += np.dot(daraw, hidden[t-1].T)
            self.daa_enc = np.dot(self.waa_enc.T, daraw)
        pass
    
    def update_params(self):
        for d in [self.dwax_enc,self.dwaa_enc,self.dwba_enc,self.dwax_dec,self.dwaa_dec,self.dwba_dec,self.dway_dec,self.dwby_dec]:
            np.clip(d,-1,1,out=d)  #gradient clipping to avoid  gradient vanishing/explotion
        
        self.wax_enc-=self.lr*self.dwax_enc
        self.waa_enc-=self.lr*self.dwaa_enc
        self.wba_enc-=self.lr*self.dwba_enc
        self.wax_dec-=self.lr*self.dwax_dec
        self.waa_dec-=self.lr*self.dwaa_dec
        self.wba_dec-=self.lr*self.dwba_dec
        self.way_dec-=self.lr*self.dway_dec
        self.wby_dec-=self.lr*self.dwby_dec
    