# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:50:11 2017

@author: Pranjal Paliwal
"""
import numpy as np
from numpy import genfromtxt
import pandas as pd

#sigmoid function
def sig_func(x):
    return 1/(1+np.exp(-x))
#sigmoid derivative
def sig_deriv(x):
    return sig_func(x)*(1 - sig_func(x))

#Creating Dataset 
"""seeds_data = pd.read_csv("seeds_binary.csv")
dataset = genfromtxt('seeds_binary.csv',delimiter=',')
feat = dataset[:,0:6]
trget = dataset[:,7,None]
input_neurons = feat.shape[1] + 1 
hidden_layerN = 3 
output_layerN = 1
X = np.insert(feat,0,np.ones((1,feat.shape[0])),1)
#print X
y = trget"""
#---------------DUMMY DATASET-------------------------
feat = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])
input_neurons = feat.shape[1] 
hidden_layerN = 3 
output_layerN = 1
X = np.insert(feat,0,np.ones((1,feat.shape[0])),1)
#----------------------------------------


#Initialising weight matrices
b12 = np.random.randn(1,hidden_layerN)
b23 = np.random.randn(1,output_layerN)
#print b12 , b23
W12 = np.random.randn(input_neurons,hidden_layerN)
W23 = np.random.randn(hidden_layerN,output_layerN)
W12 = np.insert(W12,0,b12,0)
W23 = np.insert(W23,0,b23,0)

for i in range(1):
    #Forward Pass
    h_in = np.dot(X,W12)
    h_out = sig_func(h_in)
    h_out = np.insert(h_out,0,np.ones((1,h_out.shape[0])),1)
    o_in = np.dot(h_out,W23)
    o_out = sig_func(o_in)
    
    
    #Backward Pass
    Err = y - o_out
  
    
    
    
    
    
        
    
    
    







