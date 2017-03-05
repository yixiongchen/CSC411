from pylab import *
import numpy as np
from matplotlib.pyplot import *
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
 
 
def grad_descent(df, x, y, init_w, b, alpha):
    EPS = 1e-15   #EPS = 10**(-5)
    max_iter = 5000
    iter  = 0
    count = 0;
    t = vstack((b, init_w)); 
    prev_t = t-10*EPS;
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        if(iter % 500 == 0):
            print("iteration: ", iter);
            filename ='theta_'+str(count)+'.txt'
            np.savetxt(filename, t)
            count += 1;
        prev_t = t.copy()
        t -= alpha*df(x, y, t).T
        iter += 1

 
#vectorized gradient function
def df(x, y, w):
    #get softmax
    p = compute_network(x, w);
    d = dot((p - y), x.T);
    return d;
    
    
def f(y, x, w):
    p1 = compute_network(x, w);
    return -sum(y*log(p1)) 
         
#function test_performance with theta
def test_performance(M, theta, test_list):
    total_size = 0;
    count = 0;
    for index in range(0, len(test_list)):
        for image in M[test_list[index]]:
            total_size+=1
            image = array([image]);
            image = vstack((ones((1, image.shape[0])), image.T))
            if(argmax(compute_network(image, theta)) == index):
                count+=1;
    performance = count /total_size;
    print("count is ", count);
    print("total_size is ", total_size);
    print(performance);
    return performance;


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

# """
# PART 1
# Describe the dataset of digits. In your report, include 10 images of each of the digits.
# """
# 
# image_list = ["train0","train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]
# f,ax = plt.subplots(10,10)
# for j in range(10):
#     for i in range(10):
#         ax[j][i].imshow(M[image_list[j]][i].reshape((28,28)), cmap=cm.gray)
# plt.show() 


"""
PART 2
Implement a function that computes the network below.

#x is the matrix 785x1; w is the matrix 785x10;, b is the
#matrix 10x1 
"""
def compute_network(x, w):
    #linear combination
    y = dot(w.T, x);
    #softmax
    output=exp(y)/tile(sum(exp(y),0), (len(y),1))
    return output;


"""
PART 3

"""
###part(b)
#Write vectorized code that computes the gradient with respect to #the cost function. Check that the gradient was computed correctly #by approximating the gradient at several coordinates using finite #differences. Include the code for computing the gradient in #vectorized form in your report.


    
#test difference
part3_x = array([[1,1,2,3,4]]).T
part3_w =array([[.1,.1,.1,.1,.1],[.1,.1,.1,.1,.1], [.1,.1,.1,.1,.1]])
part3_y = array([[1], [0], [0]])

h = [0.0000001, 0.0000002, 0.0000003, 0.0000004]
for index in h:
    w1 = array([[.1+index,.1,.1,.1,.1],[.1,.1,.1,.1,.1], [.1,.1,.1,.1,.1]])
    finite = (f(part3_y, part3_x, w1.T) - f(part3_y, part3_x, part3_w.T)) / index; 
    gradient = df(part3_x, part3_y, part3_w.T)
    difference = abs(finite - gradient[0][0]);
    print("difference between gradient function and finite difference is", difference);





"""
Part 4
Train the neural network you constructed. Plot the learning curves. Display the weights going into each of the output units. Describe the details of your optimization procedure.

"""

#read all train example;
train_list = ["train0","train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]

test_list =  ["test0","test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9"]

number_length = [];
part4_x = M[train_list[0]];

part4_y1 = array([[1] + [0] * 9]);
part4_y2 = array([[0] * 1 + [1] + [0] * 8]);
part4_y3 = array([[0] * 2 + [1] + [0] * 7]);
part4_y4 = array([[0] * 3 + [1] + [0] * 6]);
part4_y5 = array([[0] * 4 + [1] + [0] * 5]);
part4_y6 = array([[0] * 5 + [1] + [0] * 4]);
part4_y7 = array([[0] * 6 + [1] + [0] * 3]);
part4_y8 = array([[0] * 7 + [1] + [0] * 2]);
part4_y9 = array([[0] * 8 + [1] + [0] * 1]);
part4_y10 = array([[0] * 9 + [1]]);

part4_y = part4_y1;

y_sets = [part4_y1, part4_y2, part4_y3, part4_y4, part4_y5, part4_y6, part4_y7, part4_y8, part4_y9, part4_y10]

b = array([[0.]*10])
part4_w = array([[0.]*10]*784)

number_length.append(part4_x.shape[0]);

for i in range(number_length[0]-1):
    part4_y = vstack((part4_y, part4_y1))

for number in range(1,len(train_list)):
    
    #construct x
    part4_x =vstack((part4_x, M[train_list[number]]));
    #construct y
    number_length.append(M[train_list[number]].shape[0]);
    
    for j in range(number_length[number]):
        part4_y = vstack((part4_y, y_sets[number]))


part4_x= vstack((ones((1, part4_x.shape[0])), part4_x.T))

#get a list of theta
alpha = 0.00000001;
grad_descent(df, part4_x, part4_y.T, part4_w, b, alpha);
    


result_for_test = [];
result_for_training = [];
theta_list = []
#performance for test sets
#performance for training sets
for i in range(0, 11):
    iteration = i * 500;
    filename = "theta_"+str(i)+".txt";
    theta = np.loadtxt(filename);
    theta_list.append(theta);
    print("test performance for iteration "+str(iteration)+" is");
    print("test sets:")
    result_for_test.append(test_performance(M,theta,test_list));
    print("train sets:")
    result_for_training.append(test_performance(M,theta,train_list))
    print("\n\n\n\n\n")


#draw learning curve
x_aix = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000];
#graph for act
plot(x_aix, result_for_test, "r", label = "Test set")
plot(x_aix, result_for_training, "b", label = "Training set")
legend(loc = 1)
xlim([0, 6000])
ylim([0, 1])
show();


#display theta
display_theta = theta_list[len(theta_list) -1].T;
image_title = 0;
for image in display_theta:
    #construct image array
    image_title += 1;
    image_array = [];
    subarray = []
    count = 0;
    for item in range(1, 785):
        count += 1;
        subarray.append(image[item]);
        if(count == 28):
            image_array.append(subarray);
            count = 0;
            subarray=[];
    part4_image = array(image_array);
    imsave("part4_"+ str(image_title)+".jpg", part4_image);
        




    

    
        
        
    























