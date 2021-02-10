# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 20:01:19 2020

@author: 91877
"""
import numpy as np
from Layer import Layer
from Activation import Activation
from matplotlib import pyplot as plt
import pickle

class MLP:
        

    def forward_propagate(self,net:Layer,X):
        net.params['a0']=X
        #Calculate the activations for every  layers
        for l in range(1, net.layers):
            net.params['z' + str(l)] = np.add(np.dot(net.params['W' + str(l)], net.params['a' + str(l - 1)]), net.params['b' + str(l)])
            if(net.activation[l-1]=='relu'):
                net.params['a' + str(l)] = Activation.Relu(net.params['z' + str(l)])
            elif(net.activation[l-1]=='tanh'):
                net.params['a' + str(l)] = Activation.Tanh(net.params['z' + str(l)])
            elif(net.activation[l-1]=='linear'):
                net.params['a' + str(l)] = Activation.Linear(net.params['z' + str(l)])
            else:
                net.params['a' + str(l)] = Activation.Sigmoid(net.params['z' + str(l)])
        return net
        
    def compute_cost(self,net, y):
        #compute the cost error
        m=y.shape[1]
        cost = 0.5/m *np.sum((y-net.params['a'+str(net.layers-1)])**2)
        cost=np.squeeze((cost))        
        return cost
    
    def compute_grads(self,net1,net2, y):
        #compuete gradiants
        L=net2.layers
        m=y.shape[1]
        net2.grads['dz' + str(L-1)] = net2.params['a' + str(L-1)] - y
        #dWlayers
        net2.grads['dW' + str(L-1)] = 1/m*np.dot(net2.grads['dz' + str(L-1)], net2.params['a' + str(L-2)].T)
        #dbL
        net2.grads['db' + str(L-1)] = 1/m*np.sum(net2.grads['dz' + str(L-1)],axis=1,keepdims=True)

        #Partial grads of the cost function with respect to z[layers], W[layers] and b[layers]
        for l in range(net2.layers-2, 0, -1):
            if(net2.activation[l]=='relu'):
                net2.grads['dz' + str(l)] = np.dot(np.transpose(net2.params['W' + str(l + 1)]), net2.grads['dz' + str(l + 1)])*Activation.Relu(net2.params['z' + str(l)],derivative=True)
            elif(net2.activation[l]=='tanh'):
                net2.grads['dz' + str(l)] = np.dot(np.transpose(net2.params['W' + str(l + 1)]), net2.grads['dz' + str(l + 1)])*Activation.Tanh(net2.params['z' + str(l)],derivative=True)
            elif(net2.activation[l]=='linear'):
                net2.grads['dz' + str(l)] = np.dot(np.transpose(net2.params['W' + str(l + 1)]), net2.grads['dz' + str(l + 1)])*Activation.Linear(net2.params['z' + str(l)],derivative=True)
            else:
                net2.grads['dz' + str(l)] = np.dot(np.transpose(net2.params['W' + str(l + 1)]), net2.grads['dz' + str(l + 1)])*Activation.Sigmoid(net2.params['z' + str(l)],derivative=True)
            net2.grads['dW' + str(l)] = 1/m * (np.dot(net2.grads['dz' + str(l)], np.transpose(net2.params['a' + str(l - 1)])))
            net2.grads['db' + str(l)] = 1 /m * (np.sum(net2.grads['dz'+str(l)],axis = 1,keepdims = True))
        #compute the gradient of network1 through network 2
        weight_2_1=net2.params['W1'][:,0:net1.nodes[-1]]
        if(net1.activation[0]=='relu'):
            net1.grads['dz1']=np.dot(weight_2_1.T,net2.params['z1'])*Activation.Relu(net1.params['z1'],derivative=True)
        elif(net1.activation[0]=='tanh'):
            net1.grads['dz1']=np.dot(weight_2_1.T,net2.params['z1'])*Activation.Tanh(net1.params['z1'],derivative=True)
        elif(net1.activation[0]=='linear'):
            net1.grads['dz1']=np.dot(weight_2_1.T,net2.params['z1'])*Activation.Linear(net1.params['z1'],derivative=True)
        else:
            net1.grads['dz1']=np.dot(weight_2_1.T,net2.params['z1'])*Activation.Relu(net1.params['z1'],derivative=True)
        net1.grads['dW1']=1/m * np.dot(net1.grads['dz1'], np.transpose(net1.params['a0']))
        net1.grads['db1']=1 / m * (np.sum(net1.grads['dz1'],axis = 1,keepdims = True))
        pass
    
    def update_params(self,net,lr1):
        
        for l in range(1, net.layers):
            net.params['W' + str(l)] -= lr1*net.grads['dW' + str(l)]
            net.params['b' + str(l)] -= lr1*net.grads['db' + str(l)]
        pass
        
model=MLP()   
def saveweight(net1,net2):
    params=[net1,net2]
    with open('weights.dat','wb') as f:
        pickle.dump(params,f)
    print('weights saved successfully')
    pass
    


 
def Train(X,Xnet2,Y,net1,net2,epochs,lr1,lr2,printcost=False):
    m=X.shape[1]
    costs=[]
    for i in range(0,epochs):
        
        net1=model.forward_propagate(net1,X)
        prop=np.concatenate([net1.params['a1'],Xnet2])
        net2=model.forward_propagate(net2,prop)
        cost=model.compute_cost(net2,Y)
        costs.append(cost)
        model.compute_grads(net1,net2,Y)
        if(i<epochs/2):
            model.update_params(net1,lr1)
        model.update_params(net2,lr2)

        if printcost and i % 10 == 0:
            costs.append(cost)
            print('cost:' + str(cost))
        
     # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Trainingloss")
    plt.show()
    saveweight(net1,net2)
    pass

def Predict(X,Xnet2,net1,net2):
    #predict with the cross validation set
    net_1=model.forward_propagate(net1,X)
    prop=np.concatenate([net_1.params['a1'],Xnet2])
    net_2=model.forward_propagate(net2,prop)
    return net2.params['a'+str(net2.layers-1)]


def Test(X,X_net2,test_days,net1,net2):
    #evaluation function to test with test set
    y=np.zeros((X_net2.shape[0],X_net2.shape[1]))
    m=X.shape[1]
    i=1
    size=209 #number of countries
    while(i<17):   #17 days are in test set
        X_test_batch_net1=X[:,(i-1)*size:i*size]
        X_test_batch_net2=X_net2[:,(i-1)*size:i*size]
        test_day=test_days[:,(i-1)*size:i*size]
        X_test_batch_net2 =np.concatenate([test_day,X_test_batch_net2])
        net_1=model.forward_propagate(net1, X_test_batch_net1)
        prop=np.concatenate([net_1.params['a1'],X_test_batch_net2])
        net_2=model.forward_propagate(net2,prop)
        y[:,(i-1)*size:i*size]=net_2.params['a'+str(net_2.layers-1)]
        X_net2[:,i*size:(i+1)*size]=y[:,(i-1)*size:i*size]
        i+=1
        
    return y
        
    
    
    
    

        
