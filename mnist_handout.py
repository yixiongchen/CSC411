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
import random
from scipy.misc import imsave
import scipy.stats
import pickle
import hashlib
import os
from scipy.io import loadmat
import fnmatch
import shutil
import pandas as pd
from numpy.linalg import norm

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

import tensorflow as tf



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
    max_iter = 30000
    iter  = 0
    count = 0;
    t = vstack((b, init_w)); 
    prev_t = t-10*EPS;
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        if(iter % 300 == 0):
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
 
 
def compute_network(x, w):
    #linear combination
    y = dot(w.T, x);
    #softmax
    output=exp(y)/tile(sum(exp(y),0), (len(y),1))
    #print("netwrokoutput:", output);
    return output;

        
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


    

### logistic regression function
def part5_log_grad_descent(df, x, y, init_w, alpha):
    EPS = 1e-15   #EPS = 10**(-5)
    max_iter = 4000
    iter  = 0
    count = 0;
    t = init_w; 
    prev_t = t-10*EPS;
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t).T
        iter += 1
    return t;
 
###linear regression functions
#hypothesis functions
def linear_h(x,theta):
    return dot(theta,x)

# cost function
def linear_f(x, y, theta, m):
    #x = vstack( (ones((1, x.shape[1])), x))
    return sum((y-dot(theta.T,x)) ** 2) / (2*m)

#gradient function
def linear_df(x, y, theta, m):
    #x = vstack( (ones((1, x.shape[1])), x))
    return -sum((y-dot(theta.T, x))*x, 1) / m 
 
#gradient decent        
def part5_linear_grad_decent(f, df, x, y, init_t, alpha, m):
    EPS = 1e-8  
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 3000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()

        t -= alpha*df(x, y, t, m)
        
        iter += 1
    return t


def mini_batch(Act_M, lamda):
    x = tf.placeholder(tf.float32, [None, 1024])
    nhid = 1000
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    #activation function
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])
    
    lam = lamda
    decay_penalty=lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    train_step = tf.train.AdamOptimizer(0.0008).minimize(reg_NLL)
    init = tf.initialize_all_variables() 
    sess = tf.Session()
    sess.run(init)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_test(Act_M)
    valid_x, valid_y = get_validation(Act_M)
    
    test_cost_result = [];
    x_aix=[]
    for i in range(1000):
        batch_xs, batch_ys = get_train_batch(Act_M, 60);
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 10 == 0:
            x_aix.append(i);
            print("iteration:", i);            
            t = sess.run(accuracy, feed_dict={x: test_x, y_: test_y});
            # get cost of test sets
            test_x = test_x.astype(float32);
            layer1 = tf.nn.tanh(tf.matmul(test_x, W0)+b0);
            layer2 = tf.matmul(layer1, W1)+b1;
            newy = tf.nn.softmax(layer2);
            newy_= test_y.astype(float32);
            reg_NLL = -tf.reduce_sum(newy_*tf.log(newy))+decay_penalty;
            test_cost_result.append(sess.run(reg_NLL)/test_x.shape[0]);
            
    return x_aix, test_cost_result;

        
        
    


##############



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




"""
PART 3

"""
###part(b)
#Write vectorized code that computes the gradient with respect to #the cost function. Check that the gradient was computed correctly #by approximating the gradient at several coordinates using finite #differences. Include the code for computing the gradient in #vectorized form in your report.


    
#test difference
part3_x = array([[1,1,2,3,4],[1,2,3,4,6]]).T
part3_w =array([[.1,.1,.1,.1,.1],[.1,.1,.1,.1,.1], [.1,.1,.1,.1,.1]])
part3_y = array([[1,0], [0,1], [0,0]])

h = [0.0000001, 0.0000002, 0.0000003, 0.0000004]
for index in h:
    w1 = array([[.1+index,.1,.1,.1,.1],[.1,.1,.1,.1,.1], [.1,.1,.1,.1,.1]])
    finite = (f(part3_y, part3_x, w1.T) - f(part3_y, part3_x, part3_w.T)) / index; 
    gradient = df(part3_x, part3_y, part3_w.T)
    difference = abs(finite - gradient[0][0]);
    print("difference between gradient function and finite difference is", difference);





# """
# Part 4
# Train the neural network you constructed. Plot the learning curves. Display the weights going into each of the output units. Describe the details of your optimization procedure.
# 
# """
# 
# #read all train example;
# train_list = ["train0","train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]
# 
# test_list =  ["test0","test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9"]
# 
# number_length = [];
# part4_x = M[train_list[0]];
# 
# part4_y1 = array([[1] + [0] * 9]);
# part4_y2 = array([[0] * 1 + [1] + [0] * 8]);
# part4_y3 = array([[0] * 2 + [1] + [0] * 7]);
# part4_y4 = array([[0] * 3 + [1] + [0] * 6]);
# part4_y5 = array([[0] * 4 + [1] + [0] * 5]);
# part4_y6 = array([[0] * 5 + [1] + [0] * 4]);
# part4_y7 = array([[0] * 6 + [1] + [0] * 3]);
# part4_y8 = array([[0] * 7 + [1] + [0] * 2]);
# part4_y9 = array([[0] * 8 + [1] + [0] * 1]);
# part4_y10 = array([[0] * 9 + [1]]);
# 
# part4_y = part4_y1;
# 
# y_sets = [part4_y1, part4_y2, part4_y3, part4_y4, part4_y5, part4_y6, part4_y7, part4_y8, part4_y9, part4_y10]
# 
# b = array([[0.]*10])
# part4_w = array([[0.]*10]*784)
# 
# number_length.append(part4_x.shape[0]);
# 
# for i in range(number_length[0]-1):
#     part4_y = vstack((part4_y, part4_y1))
# 
# for number in range(1,len(train_list)):
#     
#     #construct x
#     part4_x =vstack((part4_x, M[train_list[number]]));
#     #construct y
#     number_length.append(M[train_list[number]].shape[0]);
#     
#     for j in range(number_length[number]):
#         part4_y = vstack((part4_y, y_sets[number]))
# 
# 
# part4_x= vstack((ones((1, part4_x.shape[0])), part4_x.T))
# 
# #get a list of theta
# alpha = 0.0000000001;
# grad_descent(df, part4_x, part4_y.T, part4_w, b, alpha);
#     
# 
# 
# result_for_test = [];
# result_for_training = [];
# theta_list = []
# #performance for test sets
# #performance for training sets
# for i in range(0, 11):
#     iteration = i * 500;
#     filename = "theta_"+str(i)+".txt";
#     theta = np.loadtxt(filename);
#     theta_list.append(theta);
#     print("test performance for iteration "+str(iteration)+" is");
#     print("test sets:")
#     result_for_test.append(test_performance(M,theta,test_list));
#     print("train sets:")
#     result_for_training.append(test_performance(M,theta,train_list))
#     print("\n\n\n\n\n")
# 
# 
# #draw learning curve
# x_aix = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000];
# #graph for act
# plot(x_aix, result_for_test, "r", label = "Test set")
# plot(x_aix, result_for_training, "b", label = "Training set")
# legend(loc = 1)
# xlim([0, 3000])
# ylim([0, 1])
# show();
# 
# 
# #display theta
# display_theta = theta_list[len(theta_list) -1].T;
# image_title = 0;
# for image in display_theta:
#     #construct image array
#     image_title += 1;
#     image_array = [];
#     subarray = []
#     count = 0;
#     for item in range(1, 785):
#         count += 1;
#         subarray.append(image[item]);
#         if(count == 28):
#             image_array.append(subarray);
#             count = 0;
#             subarray=[];
#     part4_image = array(image_array);
#     imsave("part4_"+ str(image_title)+".jpg", part4_image);
#         




"""
Part 5
The advantage of using multinomial logistic regression (i.e., the network you constructed here) over using linear regression as in Project 1 is that the cost function isn’t overly large when the actual outputs are very far away from the target outputs. That causes the network to not adjust the weights too much because of a single training case that causes the outputs to be very large. You should come up with a dataset where the performance on the test set is better when you use multinomial logistic regression. Do this by generating training and test sets similarly to how we did it in lecture . Show that the performance using multinomial logistic regression (on the test set) is substantially better. Explain what you did and include code to clarify your point. Clearly explain why your experiment illustrates your point.
"""

def test_part5_performance(N, sigma):
    
    test_result= [];
    theta = array([-3, 1.5])
    
    #generate train sets
    part5_train_x_raw = 100*(np.random.random((N))-.5)
    part5_train_x = vstack((ones_like(part5_train_x_raw),part5_train_x_raw,))
    part5_train_y = dot(theta, part5_train_x) + scipy.stats.norm.rvs(scale=
    sigma,size=N)
    
    #generate test sets
    part5_test_x_raw = 100*(random.random((N))-.5)
    part5_test_x = vstack(( ones_like(part5_test_x_raw),part5_test_x_raw,))
    part5_test_y = dot(theta, part5_test_x) + scipy.stats.norm.rvs(scale=
    sigma,size=N)
    
    ## linear regression 
    # reset train data output as 0 and 1 for y
    linear_train =[]
    for y in part5_train_y:
        if y > 0:
            linear_train.append(1);
        else:
            linear_train.append(0);
        linear_train_y = array(linear_train);
    
    #reset test data output as 0 and 1 for y
    linear_test = []
    for y in part5_test_y:
        if y > 0:
            linear_test.append(1);
        else:
            linear_test.append(0);
        linear_test_y = array(linear_test);
    
    alpha = 0.0000000001;
    # generate theta by linear regression
    linear_init_theta = array([0., 0.])
    linear_theta = part5_linear_grad_decent(linear_f, linear_df,
    part5_train_x, linear_train_y, linear_init_theta, alpha, N)
    
    ##logistic regression
    #reset train data output as [0, 1] and [1, 0] for y
    logistic_train_y =[];
    for y in part5_train_y:
        sublist = [];
        if y > 0:
            sublist.append(0);
            sublist.append(1);
        else:
            sublist.append(1);
            sublist.append(0);
        logistic_train_y.append(sublist);
        
    logistic_train_y=array(logistic_train_y);
    
    #reset test data output as [1, 0] and [0, 1] for y
    logistic_test_y =[];
    for y in part5_test_y:
        sublist1 = [];
        if y > 0:
            sublist1.append(0);
            sublist1.append(1);
        else:
            sublist1.append(1);
            sublist1.append(0);
        logistic_test_y.append(sublist1);
    logistic_test_y = array(logistic_test_y);
    
    #generate theta by logistic regression
    logistic_init_weight = array([[0.]*2]*2);
    logistic_theta = part5_log_grad_descent(df, part5_train_x,
    logistic_train_y.T, logistic_init_weight, alpha);
    
    
    ### test result for two methods
    count = 0;
    for i in range(0, N):
        if((linear_test_y[i] == 1) &
        (linear_h(part5_test_x.T[i],linear_theta)>=0.5)):
            count += 1;
        if((linear_test_y[i] == 0) &
        (linear_h(part5_test_x.T[i],linear_theta)<0.5)):
            count += 1;
    print("performance for linear regression: ")
    print(count/N);
    test_result.append(count/N);
    count1 = 0;
    for i in range(0, N):
        if((logistic_test_y[i][0] == 1) &
        (argmax(compute_network(part5_test_x.T[i], logistic_theta)) == 0)):
            count1 += 1;
        if((logistic_test_y[i][0] == 0) &
        (argmax(compute_network(part5_test_x.T[i], logistic_theta)) == 1)):
            count1 += 1;
    print("performance for logistic regression: ")
    print(count1/N);
    test_result.append(count1/N);
    return test_result;
    

# #test performace with size of test sets:
# N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,100]
# test_result_linear = [];
# test_result_logistic = [];
# for n in N_list:
#     result = test_part5_performance(n, 20.2);
#     test_result_linear.append(result[0]);
#     test_result_logistic.append(result[1]);
#         
# x_aix = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
# plot(x_aix, test_result_linear, "r", label = "Linear regression")
# plot(x_aix, test_result_logistic, "b", label = "multinomial logistic regression")
# legend(loc = 1)
# xlim([0, 100])
# ylim([0, 1.1])
# suptitle('Performance:  linear vs logistic ')
# xlabel('test set size')
# ylabel('correction rate')
# show();

# #test performace with sigma value:
# sigma_list = [20, 30, 40, 50, 60, 70, 80, 90,100, 120]
# test_result_linear = [];
# test_result_logistic = [];
# for sig in sigma_list:
#     result = test_part5_performance(1000, sig);
#     test_result_linear.append(result[0]);
#     test_result_logistic.append(result[1]);
#         
# x_aix = sigma_list;
# plot(x_aix, test_result_linear, "r", label = "Linear regression")
# plot(x_aix, test_result_logistic, "b", label = "multinomial logistic regression")
# legend(loc = 1)
# suptitle('Performance:  linear vs logistic ')
# xlim([0, 120])
# ylim([0, 1.1])
# xlabel('sigma')
# ylabel('correction rate')
# show();

        







"""
part 6
Backpropagation can be seen as a way to speed up the computation of the gradient. For a network with NN layers each of which contains KK neurons, determine how much faster is (fully-vectorized) Backpropagation compared to computing the gradient with respect to each weight individually. Assume that all the layers are fully-connected. Show your work. Make any reasonable assumptions (e.g., about how fast matrix multiplication can be peformed), and state what assumptions you are making.

"""

#see report







"""
part 7
I am providing the TensorFlow code for training a single-hidden-layer fully-connected network here . The training is done using mini-batches. Modify the code to classify faces of the 6 actors in Project 1. Use a fully-connected neural network with a single hidden layer. In your report, include the learning curve for the test, training, and validation sets, and the final performance classification on the test set. Include a text description of your system. In particular, describe how you preprocessed the input and initialized the weights, what activation function you used, and what the exact architecture of the network that you selected was. Experiment with different settings to produce the best performance, and report what you did to obtain the best performance.
"""

# Before executing the following codes, you must have a folder called  #'uncropped" which contains all original face images


#return greyscale image
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


#generate training , test, and validation sets
# ----
def create_sets(act):
    #remove any bad images from cropped folder
    if not os.path.exists("part7"):
        os.mkdir("part7");
        #create folder for each actor in act at first
        for i in act:
            actor_name = i.split()[1].lower();
            if not os.path.exists("part7/" + actor_name ):
                os.mkdir("part7/" + actor_name);
                #create trainning folder, validatin folder, test folder for
                #each actor 
                os.mkdir("part7/" + actor_name+ "/" + "training");
                os.mkdir("part7/" + actor_name+ "/" + "validation");
                os.mkdir("part7/" + actor_name+ "/" + "test");
                
                #create trainning set, validation set and test set for each
                #actor
                # create a list of face images that actor owns
                list_of_images = fnmatch.filter(os.listdir('cropped'),
                actor_name.lower()+'*');
                face_num = len(list_of_images);
                #generate a list of ramdom for sets
                random.seed(0);
                print(face_num);
                random_num=random.sample(range(0, face_num), 130);
                #choose the first 30 as the test set
                test_set = random_num[:30];
                #choose the next 100 numbers as trainning set
                training_set = random_num[30:120];
                #choose the next 10 numbers as validation set
                validation_set =  random_num[120:130];
                #save each image in training set into actor's trainning
                #folder
                for index in training_set:
                    im = imread("cropped/"+list_of_images[index]);
                    imsave("part7/"+ actor_name+
                    "/"+"training/"+list_of_images[index], im);
                #save each image in validation set into actor's test folder
                for index in validation_set:
                    im = imread("cropped/"+list_of_images[index]);
                    imsave("part7/"+ actor_name+
                    "/"+"validation/"+list_of_images[index], im);
                
                #save each image in validation set into actor's validation
                #folder
                for index in test_set:
                    im = imread("cropped/"+list_of_images[index]);
                    imsave("part7/"+ actor_name+
                    "/"+"test/"+list_of_images[index], im);
    
    
# removed bad images from uncropped folder and save good images in 
# cropped folder
def prepare_data():
    image_list = os.listdir('uncropped');
    count = 0;
    for image in image_list:
        a = image.split('.')[0];
        a = ''.join(i for i in a if not i.isdigit());
        a = a.title();
        flag = 0;
        try:
            im = open("uncropped/"+image, "rb").read();
        except:
            print("can not open: ", image);
            os.remove("uncropped/"+image);
        #calculate hash
        m =hashlib.sha256();
        m.update(im);
        hash = m.hexdigest();
        #print("image hash:", m);
        for line in open("subset_actors.txt"):
            if a in line:
                try:
                    #get the hashkey from original file
                    correct_hash = line.split()[6];
                    if(correct_hash == hash):
                        position = line.split()[5].split(',')
                        y1 = int(position[1])
                        y2 = int(position[3])
                        x1 = int(position[0])
                        x2 = int(position[2])
                        new_im = imread("uncropped/"+image);
                        face = new_im[y1:y2, x1:x2, :]
                        gray = rgb2gray(face)
                        cropped = imresize(gray, (32, 32));
                        color = imresize(face, (32, 32));
                        imsave("cropped/"+image, cropped);
                        imsave("colored/"+image, color);
                        flag = 1;
                        print(a);
                        break;
                except:
                    continue;
        #remove bad images
        if(flag == 0):
            count+= 1;
            print("bad image");
            os.remove("uncropped/"+image);
                
    print("bad images removed is:", count);
            
   


## prepare trainning sets , test sets and validation sets 
if (not os.path.exists("cropped")):
    os.mkdir("cropped");
    prepare_data();
    act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 
    'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    create_sets(act);
 

    
## do mini_batch for face images
#randomly select training sets 
def get_train_batch(M, N):
    n = int(N/6)
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]

    train_size = len(M[train_k[0]])
    for k in range(6):
        train_size = len(M[train_k[k]])
        idx = array(np.random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    
#get all test sets 
def get_test(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        
# get all training sets 
def get_train(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s


# get all validation sets
def get_validation(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, 6))
    
    valid_k =  ["validation"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s 
 
 
 

#get train sets test sets and validation sets
def grey_image_setting():
    Act_M={};
    actor =['drescher', 'ferrera', 'chenoweth', 
    'baldwin', 'hader', 'carell']
    #get training sets
    for i in range(0, len(actor)):
        player = os.listdir("part7/"+actor[i]+"/training");
        key = "train"+str(i);
        train_list =[];
        for image in player:
            sub=[]
            im = imread("part7/"+actor[i]+"/training/"+image);
            for elem in im:
                for second in elem:
                    sub.append(second);
            train_list.append(sub);
        train_list = array(train_list);
        Act_M[key] = train_list
    
    #get test sets
    for i in range(0, len(actor)):
        player = os.listdir("part7/"+actor[i]+"/test");
        key = "test"+str(i);
        test_list =[];
        for image in player:
            sub=[]
            im = imread("part7/"+actor[i]+"/test/"+image);
            for elem in im:
                for second in elem:
                    sub.append(second);
            test_list.append(sub);
        test_list = array(test_list);
        Act_M[key] = test_list
    
    #get validation sets
    for i in range(0, len(actor)):
        player = os.listdir("part7/"+actor[i]+"/validation");
        key = "validation"+str(i);
        valid_list =[];
        for image in player:
            sub=[]
            im = imread("part7/"+actor[i]+"/validation/"+image);
            for elem in im:
                for second in elem:
                    sub.append(second);
            valid_list.append(sub);
        valid_list = array(valid_list);
        Act_M[key] = valid_list
    
    return Act_M;



#test performance
Act_M = grey_image_setting();

x = tf.placeholder(tf.float32, [None, 1024])
nhid = 300
W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


#activation function
layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])

#lam = 1
lam = 0.0000
decay_penalty=lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
train_step = tf.train.AdamOptimizer(0.0008).minimize(reg_NLL)
init = tf.initialize_all_variables() 
sess = tf.Session()
sess.run(init)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(Act_M)
valid_x, valid_y = get_validation(Act_M)


if os.path.exists("snapshot"):
    shutil.rmtree("snapshot");
if not os.path.exists("snapshot"):
    os.mkdir("snapshot");


    
#save performance resut
#part8 error=[]
test_result =[]
train_result = [];
valid_result = [];
test_cost_result = [];
train_cost_result= [];
valid_cost_result =[];

x_aix=[]
for i in range(1001):
    
    batch_xs, batch_ys = get_train_batch(Act_M, 60);
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 10 == 0:
        
        x_aix.append(i);
        print("iteration:", i);
        
        t = sess.run(accuracy, feed_dict={x: test_x, y_: test_y});
        test_result.append(t);
        print("test sets: ",t);

        v = sess.run(accuracy, feed_dict={x: valid_x, y_:valid_y});
        valid_result.append(v);
        print("validation set: ", v);
        
        batch_xs, batch_ys = get_train(Act_M)
        t2 = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        train_result.append(t2);
        print("training set: ", t2);
        print ("Penalty:", sess.run(decay_penalty))
        
        #get cost of valid sets
        valid_x = valid_x.astype(float32);
        layer1 = tf.nn.tanh(tf.matmul(valid_x, W0)+b0);
        layer2 = tf.matmul(layer1, W1)+b1;
        newy = tf.nn.softmax(layer2);
        newy_= valid_y.astype(float32);
        reg_NLL = -tf.reduce_sum(newy_*tf.log(newy))+decay_penalty;
        valid_cost_result.append(sess.run(reg_NLL)/valid_x.shape[0]);
        print("cost for valid is:", sess.run(reg_NLL));
        
        
        
        # get cost of test sets
        test_x = test_x.astype(float32);
        layer1 = tf.nn.tanh(tf.matmul(test_x, W0)+b0);
        layer2 = tf.matmul(layer1, W1)+b1;
        newy = tf.nn.softmax(layer2);
        newy_= test_y.astype(float32);
        reg_NLL = -tf.reduce_sum(newy_*tf.log(newy))+decay_penalty;
        test_cost_result.append(sess.run(reg_NLL)/test_x.shape[0]);
        print("cost for test is:", sess.run(reg_NLL));
        
        # get cost of training sets
        train_x = batch_xs.astype(float32);
        layer1 = tf.nn.tanh(tf.matmul(train_x, W0)+b0);
        layer2 = tf.matmul(layer1, W1)+b1;
        y = tf.nn.softmax(layer2);
        reg_NLL = -tf.reduce_sum(batch_ys*tf.log(y))+decay_penalty;
        train_cost_result.append(sess.run(reg_NLL)/train_x.shape[0]);
        print("cost for train is:", sess.run(reg_NLL));
        
        
        print("\n")
        snapshot = {}
        snapshot["W0"] = sess.run(W0)
        snapshot["W1"] = sess.run(W1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        pickle.dump(snapshot, 
        open("snapshot/new_snapshot"+str(i)+".pkl", 
        "wb"))


#error rate curve
figure(1)
plot(x_aix, valid_cost_result, "g", label = "validation")
plot(x_aix, test_cost_result, "r", label = "test")
plot(x_aix, train_cost_result, "b", label = "training")
legend(loc = 1)
suptitle('Error/cost curve')
xlim([0, 1000])
ylim([0, 3])
xlabel('iteration')
ylabel('cost')
savefig("part7_1")
show();

        
# typical learning curve
figure(2)
plot(x_aix, test_result, "r", label = "test")
plot(x_aix, train_result, "b", label = "training")
plot(x_aix, valid_result, "g", label = "validation")
legend(loc = 4)
suptitle('Typical Learning Curve')
xlim([0, 1000])
ylim([0, 1.1])
xlabel('iteration')
ylabel('correction rate')
savefig("part7_2")
show();







"""
PART 8
You are not required to use regularization in Part 7 (although that would produce nicer weight visualization). Find a scenario where using regularization is necessary in the context of face classification, and find the best regularization parameter λ .


Suppose we set the number of hidden unit = 1000, and generate the w0, w1, b0, b1 with larger stand deviation = 0.01. Same train sets as before
Case 1: set lamda = 0, After 800 iteration, the cost of test sets dramatically increase, it is overfitting 
Case 2: set lambda > 0, The lamba provide penalty on cost function which limit the size of hidden units and wights, it prevent overfitting by stop iterations before large overfitting.

"""
# Act_M =  grey_image_setting();
# x_aix, lamda_cost = mini_batch(Act_M, 1);
# x_aix, nolamda_cost = mini_batch(Act_M, 0);
# 
# 
# figure(3)
# plot(x_aix, lamda_cost, "r", label="Lambda penalty")
# plot(x_aix, nolamda_cost, "b", label="No penalty")
# legend(loc = 1)
# suptitle('Learning curve')
# xlim([0, 1000])
# ylim([0, 3])
# xlabel('iteration')
# ylabel('cost')
# savefig("part8")
# show();






"""
PART 9 (10 pts)

Select two of the actors, and visualize the weights of the hidden units that are useful for classifying input photos as those particular actors. Explain how you selected the hidden units.

A sample visualization is below.
"""

# first load snapshot from snapshot/new_snapshot3000.pkl
# distinguish drescher' and 'ferrera 
snapshot =  pickle.load(open("snapshot/new_snapshot1000.pkl", "rb"), encoding="latin1");
W0 = snapshot["W0"];
b0 = snapshot["b0"];
W1 = snapshot["W1"];
b1 = snapshot["b1"];

test_x = W0.T;

firstlayer = tf.nn.tanh(tf.matmul(test_x, W0)+b0)
secondlayer = tf.matmul(firstlayer, W1)+b1
prob =  tf.nn.softmax(secondlayer)
output = sess.run(prob);
drescher = -1;
ferrera = -1;
hidden_list_1 = [];
hidden_list_2 = [];
for i in range(output.shape[0]):
    #check drescher
    if argmax(output[i]) == 0:
        drescher = i;
        hidden_list_1.append(i);
    #check ferrera
    if argmax(output[i]) == 1:
        ferrera  = i;
        hidden_list_2.append(i);

print("total drescher: ", hidden_list_1);
print("total ferrera: ", hidden_list_2);



#visualize the weights for the hidden layer corresponding to the actor 
for num in range(5):
    image_drescher = [];
    image_ferrera = [];
    subarray1 = [];
    subarray2 = [];
    count = 0;
    weight_for_drescher = test_x[hidden_list_1[num]];
    weight_for_ferrera = test_x[hidden_list_2[num]];
    for i in range(1024):
        subarray1.append(weight_for_drescher[i]);
        subarray2.append(weight_for_ferrera[i]);
        count += 1;
        if count == 32:
            image_drescher.append(subarray1);
            image_ferrera.append(subarray2);
            subarray1 =[];
            subarray2 =[];
            count = 0;
    
    image_drescher = array(image_drescher);
    image_ferrera = array(image_ferrera);

    imsave("part9_drescher"+str(num)+".jpg", image_drescher);
    imsave("part9_ferrerar"+str(num)+".jpg", image_ferrera);
        

















