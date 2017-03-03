from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")





def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    

#Load sample weights for the multilayer neural network
snapshot = pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1");
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################

"""
PART 1
Describe the dataset of digits. In your report, include 10 images of each of the digits.
"""

image_list = ["train0","train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]
f,ax = plt.subplots(10,10)
for j in range(10):
    for i in range(10):
        ax[j][i].imshow(M[image_list[j]][i].reshape((28,28)), cmap=cm.gray)
plt.show() 


"""
PART 2
Implement a function that computes the network below.

#x is the matrix 784x1; w is the matrix 784x9;, b is the
#matrix 9x1 
"""
def compute_network(x, w, b):
    #linear combination
    y = dot(w.T, x) + b;
    #softmax
    output=exp(y)/tile(sum(exp(y),0), (len(y),1))
    return output;
    

"""
PART 3

"""
###part(b)
#Write vectorized code that computes the gradient with respect to #the cost function. Check that the gradient was computed correctly #by approximating the gradient at several coordinates using finite #differences. Include the code for computing the gradient in #vectorized form in your report.

def df(x, y, w, b):
    #get softmax
    p = computer_network(x, w, b);
    d = dot((p - y), x.T);
    return d;
    
#test difference















