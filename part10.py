################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
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
import random
import scipy.stats
import pickle
import tensorflow as tf
from caffe_classes import class_names
import hashlib
import os
from scipy.io import loadmat
import fnmatch
import shutil
import pandas as pd
from numpy.linalg import norm
from scipy.misc import imsave




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
                        color = imresize(face, (227, 227));
                        imsave("colored/"+image, color);
                        break;
                except:
                    continue;



# create 227 x 227 x 3 image
def create_sets(act):
    #remove any bad images from cropped folder
    if not os.path.exists("part10"):
        os.mkdir("part10");
        #create folder for each actor in act at first
        for i in act:
            actor_name = i.split()[1].lower();
            if not os.path.exists("part10/" + actor_name ):
                os.mkdir("part10/" + actor_name);
                #create trainning folder, validatin folder, test folder for
                #each actor 
                os.mkdir("part10/" + actor_name+ "/" + "training");
                os.mkdir("part10/" + actor_name+ "/" + "validation");
                os.mkdir("part10/" + actor_name+ "/" + "test");
                
                #create trainning set, validation set and test set for each
                #actor
                # create a list of face images that actor owns
                list_of_images = fnmatch.filter(os.listdir('uncropped'),
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
                    im = (imread("colored/"+list_of_images[index])[:,:,:3]).astype(float32);
                    #im = imresize(im, (227, 227));
                    imsave("part10/"+ actor_name+
                    "/"+"training/"+list_of_images[index], im);
                #save each image in validation set into actor's test folder
                for index in validation_set:
                    im = (imread("colored/"+list_of_images[index])[:,:,:3]).astype(float32);
                    #im = imresize(im, (227, 227));
                    imsave("part10/"+ actor_name+
                    "/"+"validation/"+list_of_images[index], im);
                
                #save each image in validation set into actor's validation
                #folder
                for index in test_set:
                    im = (imread("colored/"+list_of_images[index])[:,:,:3]).astype(float32);
                    #im = imresize(im, (227, 227));
                    imsave("part10/"+ actor_name+
                    "/"+"test/"+list_of_images[index], im);
    


# store all images in dictionary
def image_setting():
    Act_M={};
    actor =['drescher', 'ferrera', 'chenoweth', 
    'baldwin', 'hader', 'carell']
    #get training sets
    for i in range(0, len(actor)):
        player = os.listdir("part10/"+actor[i]+"/training");
        key = "train"+str(i);
        train_list =[];
        for image in player:
            im = imread("part10/"+actor[i]+"/training/"+image);
            im = im - mean(im);
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            train_list.append(im);
        train_list = array(train_list);
        Act_M[key] = train_list;
        print("ok");
    
    #get test sets
    for i in range(0, len(actor)):
        player = os.listdir("part10/"+actor[i]+"/test");
        key = "test"+str(i);
        test_list =[];
        for image in player:
            im = imread("part10/"+actor[i]+"/test/"+image);
            im = im - mean(im);
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            test_list.append(im)
        test_list = array(test_list);
        Act_M[key] = test_list
    
    #get validation sets
    for i in range(0, len(actor)):
        player = os.listdir("part10/"+actor[i]+"/validation");
        key = "validation"+str(i);
        valid_list = [];
        for image in player:
            im = imread("part10/"+actor[i]+"/validation/"+image);
            im = im - mean(im);
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            valid_list.append(im);
        valid_list = array(valid_list)
        Act_M[key] = valid_list
    
    return Act_M;




def get_train_batch(M, N):
    n = int(N/6)
    batch_xs = zeros((0, 227,227,3))
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
    batch_xs = zeros((0, 227,227,3))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        
# get all training sets 
def get_train(M):
    batch_xs = zeros((0, 227,227,3))
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
    batch_xs = zeros((0, 227,227,3))
    batch_y_s = zeros( (0, 6))
    
    valid_k =  ["validation"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s 
 
 





#############################################################################


#generate data first

if (not os.path.exists("colored")):
    os.mkdir("colored");
    #save cropped color images into a folder
    prepare_data();
        
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
#create train , test, valid
create_sets(act)
#save train, test, valid into dictionary
Act_M = image_setting();



train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 6))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]





################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)



#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = net_data["conv1"][0]
conv1b = net_data["conv1"][1]
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = net_data["conv2"][0]
conv2b = net_data["conv2"][1]
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = net_data["conv3"][0]
conv3b = net_data["conv3"][1]
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = net_data["conv4"][0]
conv4b = net_data["conv4"][1]
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


######
######




#extract the values of the activation4 of AlexNet on the face images
shape = conv4.shape.as_list()
dimension = shape[1] * shape[2] *shape[3];
conv4 = tf.reshape(conv4, [-1, dimension]);

nhid = 300;
W0 = tf.Variable(tf.random_normal([dimension, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

#hidden layers
layer1 = tf.nn.relu(tf.matmul(conv4, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)

y_ = tf.placeholder(tf.float32, [None, 6])

lam = 0.0000
decay_penalty=lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)
init = tf.initialize_all_variables() 
sess = tf.Session()
sess.run(init)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(Act_M)
valid_x, valid_y = get_validation(Act_M)



if os.path.exists("Part10snapshot"):
    shutil.rmtree("Part10snapshot");
if not os.path.exists("Part10snapshot"):
    os.mkdir("Part10snapshot");


test_result =[]
train_result = [];
valid_result = [];
x_aix=[]
for i in range(500):
    
    batch_xs, batch_ys = get_train_batch(Act_M, 60);
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 1 == 0:
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
        
        print("\n")
        snapshot = {}
        snapshot["W0"] = sess.run(W0)
        snapshot["W1"] = sess.run(W1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        pickle.dump(snapshot, 
        open("Part10snapshot/new_snapshot"+str(i)+".pkl", 
        "wb"))
 
 
    
#typical learning curve
figure(1)
plot(x_aix, test_result, "r", label = "test")
plot(x_aix, train_result, "b", label = "training")
plot(x_aix, valid_result, "g", label = "validation")
legend(loc = 4)
suptitle('Typical Learning Curve')
xlim([0, 500])
ylim([0, 1.1])
xlabel('iteration')
ylabel('correction rate')
savefig("part10_1")
show();
    
  
  

"""

Part 11

Produce an interesting visualization of the hidden units that were trained on top of the AlexNet conv4 features.

"""

#get the hidden layer units;

batch_xs, batch_ys = get_train(Act_M)
batch_xs = tf.convert_to_tensor(batch_xs.astype(float32))
#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1_in = conv(batch_xs, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')   
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'                                               
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#got it
shape = conv4.shape.as_list()
dimension = shape[1] * shape[2] *shape[3];
conv4 = tf.reshape(conv4, [-1, dimension]);
layer1 = tf.nn.relu(tf.matmul(conv4, W0)+b0)
bouns = sess.run(layer1);
imsave("part11.jpg", bouns);
 




