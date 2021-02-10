#-*- coding: utf-8 -*-
'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import random
import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
#import plot_confusion_matrix
import rnn_cell_GRU as rnn_cell
import rnn
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from EvoloPy import *
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from hmmlearn import hmm

 
#remove cpu occupation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
a=np.loadtxt("./tsfuse/lrts10sdata12fea.txt")
b=np.loadtxt("./tsfuse/lrts10slabel12fea.txt")

#train_x=a
#train_y=b
train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.2)
print(train_x.shape,test_x.shape)
m=len(train_y)
m1=len(test_y)
newli_train=np.zeros((m,4))
newli_test=np.zeros((m1,4))
for i,j in enumerate(train_y): 
   newli_train[i,4-int(j)]=1
#train_y=newli_train
for i,j in enumerate(test_y): 
   newli_test[i,4-int(j)]=1
#test_y=newli_test


       
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 256  #109 430
display_step = 200
batchid = 0


# Network Parameters
n_input = 12  #31  62 13MNIST data input (img shape: 28*28)
n_steps =  20 #50timesteps
n_hidden = 256# hidden layer num of features
n_classes = 4 # MNIST total classes (0-9 digits)




# reset graph
tf.reset_default_graph()


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def next_batch(batch_size):
    global batchid 
    if batchid+batch_size > len(train_x):
       batchid = 0
    batch_data = (train_x[batchid:min(batchid +batch_size, len(newli_train)),:])
    batch_labels = (newli_train[batchid:min(batchid + batch_size, len(newli_train)),:])
    batch_labels_1d = (train_y[batchid:min(batchid + batch_size, len(train_y))])
    batchid = min(batchid + batch_size, len(newli_train))
    return batch_data, batch_labels,batch_labels_1d
    




def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(tensor=x, shape=[-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(value=x, num_or_size_splits=n_steps, axis=0)
    # Define a lstm cell with tensorflow
    #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1)
    lstm_cell = rnn_cell.GRUCell(n_hidden)
    #lstm_cell = rnn_cell.LSTMCell(n_hidden,use_peepholes=True)
    # avoid overfitting
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    # 2 layers lstm
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell]*2)   
    # Get lstm cell output
    outputs, states = rnn.rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs[-1]





#pred batch*4d out batch*128d 
pred,out = RNN(x, weights, biases)
print(out.shape)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
#GWO optimizer
#optimizer = GWO.GWO(getattr(benchmarks, function_name),0,2,30,5,10000)

#Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#rf Session
#sess = tf.train.MonitoredSession()


# Launch the graph
with tf.Session() as sess:
    #rnn
    sess.run(init)
    #rf
    sess.run(rf_init_vars)
    tf.device('/gpu:0')
    step = 1
    # Training
    for i in range(1, num_steps + 1):
            # input out[-1] to rf classifier
            batch_x, batch_y, rf_batch_y = next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, n_steps, n_input)) 
            out128data,acc,_,loss = sess.run([out,accuracy,optimizer,cost], feed_dict={x: batch_x, y: batch_y})
            _, l, rf_acc = sess.run([rf_train_op, rf_loss_op,rf_accuracy_op], feed_dict={rf_x: out128data, rf_y: rf_batch_y})
            if i % 200 == 0 or i == 1:
                print('Step %i, Loss: %f, Acc: %f' % (i, l, rf_acc))
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))   
    #    rf test  Accuracy
    test_data = test_x.reshape((-1,n_steps, n_input))
    test_label = newli_test
    out128test,accuracy_test = sess.run([out,accuracy],feed_dict={x: test_data, y: test_label})
    print("Testing Accuracy:", accuracy_test) 
#    print("RF Test Accuracy:", sess.run(rf_accuracy_op, feed_dict={rf_x: out128test, rf_y: test_y}))



#        # heterogeneous ensemble learning LSTM+RF voting classification
#        # Get the next batch of MNIST data (only images are needed, not labels)
#        rf_batch_x,_, rf_batch_y = next_batch(batch_size)
#        _, l = sess.run([rf_train_op, rf_loss_op], feed_dict={rf_x: rf_batch_x, rf_y: rf_batch_y})
#        if i % 200 == 0 or i == 1:
#            acc = sess.run(rf_accuracy_op, feed_dict={rf_x: rf_batch_x, rf_y: rf_batch_y})
#            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
#    #    rf test  Accuracy
#    print("RF Test Accuracy:", sess.run(rf_accuracy_op, feed_dict={rf_x: test_x, rf_y: test_y}))
#    accuracyknn=0
     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#        #batch_x, batch_y = mnist.train.next_batch(batch_size)
#        rf_batch_x, batch_y, rf_batch_y= next_batch(batch_size)
#        # Reshape data to get 28 seq of 28 elements
#        batch_x = rf_batch_x.reshape((batch_size, n_steps, n_input))
#        #rnnpred = sess.run(logits_scaled, feed_dict={x: batch_x, y: batch_y})
#        #rnnout = sess.run(out, feed_dict={x: batch_x, y: batch_y})
#        
#
#        """
#        #knn
#        for i in range(len(rnnout)):
#	    idx=sess.run(nn_index,feed_dict={knn_input:rnnout, knn_input_test:rnnout[i]})
#	    #dis=sess.run(dis,feed_dict={knn_input:rnnout, knn_input_test:rnnout[i]})
#            if np.argmax(batch_y[i])==np.argmax(batch_y[idx]):
#                print(rnnpred[i],'------------------')     
#                rnnpred[i]=batch_y[i]* rnnpred[i]
#                print(rnnpred[i],'------------------')     
#            #if np.argmax(batch_y[i])==np.argmax(batch_y[idx]):
#                #accuracyknn+=1
#        #print("result:%f"%(1.0*accuracyknn/len(rnnout)))
#
#        """
#
#        # Run optimization op (backprop)
#        #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, knn_rnn_pred:rnnpred})
#        sess.run([optimizer,rf_train_op, rf_loss_op,rf_accuracy_op], feed_dict={x: batch_x, y: batch_y,rf_x: rf_batch_x, rf_y: rf_batch_y})
#        if step % display_step == 0:
#            # Calculate batch accuracy
#            acc,_,_,_ = sess.run([accuracy,rf_train_op, rf_loss_op,rf_accuracy_op], feed_dict={x: batch_x, y: batch_y,rf_x: rf_batch_x, rf_y: rf_batch_y})
#            # Calculate batch loss
#            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
#            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.5f}".format(acc))
#        step += 1
#    print("Optimization Finished!") 
#
#    # Calculate accuracy for 128 mnist test images
#    test_data=test_x.reshape((-1,n_steps, n_input))
#    test_label=newli_test
#    print("Testing Accuracy:", \
#        sess.run(accuracy, feed_dict={x: test_data, y: test_label,rf_x: test_x, rf_y: test_y}))
    
    saver.save(sess, './modelcache/model.ckpt')
    
    #train_x = train_x.reshape((684, n_steps, n_input))
    #trainout=sess.run(out, feed_dict={x: train_x, y: train_y})
    #np.savetxt('./trainout.txt',trainout)
    # Calculate accuracy for 128 mnist test images
    #test_data=test_x.reshape((-1,n_steps, n_input))
    #test_label=test_y
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    #saver.save(sess, './model.ckpt')
