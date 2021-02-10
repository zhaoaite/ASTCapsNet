"""
License: Apache-2.0
Author: Aite Zhao
E-mail: aitezhao@qq.com
"""

import tensorflow as tf
import rnn_cell_GRU as rnn_cell
import rnn
import numpy as np
from config import cfg
from utils import get_batch_data
from capsLayer import CapsLayer
from sklearn import metrics  
import pickle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources




class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.n_classes=52
            self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
            self.Y = tf.one_hot(self.labels, depth=self.n_classes, axis=1, dtype=tf.float32)
            # LSTM Parameters
            self.n_input=45
            self.n_steps=45
            
            # tf Graph input
            self.lstm_x = tf.reshape(self.X, shape=(cfg.batch_size, self.n_steps, self.n_input))
            self.lstm_y = tf.reshape(self.Y, shape=(cfg.batch_size, self.n_classes))
            
            
            self.kernel_size1=13
            self.kernel_size2=9
            self.conv1_outa=self.n_input-self.kernel_size1+1
            self.conv1_outb=self.n_steps-self.kernel_size1+1
#            self.cap1_out=(self.conv1_outa-self.kernel_size+1)/2*((self.conv1_outb-self.kernel_size+1)/2)
#            self.cap1_out=int((self.conv1_outa-self.kernel_size2+1)*(self.conv1_outb-self.kernel_size2+1)*32/4)
            self.cap1_out=5408
            self.n_hidden= self.conv1_outb
            
            # Define weights
            self.lstm_weights = {
                'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
            }
            self.lstm_biases = {
                'out': tf.Variable(tf.random_normal([self.n_classes]))
            }

            if is_training:
                self.build_arch()
                self.loss()
                self._summary()
                # t_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
                self.train_c1 = self.optimizer.minimize(self.lstm_cost, global_step=self.global_step)
                self.train_c2 = self.optimizer.minimize(self.dense_cov1_cost, global_step=self.global_step)
                self.train_c3 = self.optimizer.minimize(self.dense_caps1_cost, global_step=self.global_step)
                print('end net')
            else:
                
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size,self.n_input, self.n_steps, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, self.n_classes, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')
    
    
    
    def RNN(self):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
        # Permuting batch_size and n_steps
        x = tf.transpose(self.lstm_x, [1, 0, 2])
        
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(tensor=x, shape=[-1, self.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(value=x, num_or_size_splits=self.n_steps, axis=0)
        # Define a lstm cell with tensorflow
        #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1)
        lstm_cell = rnn_cell.GRUCell(self.n_hidden)
        #lstm_cell = rnn_cell.LSTMCell(n_hidden,use_peepholes=True)
        # avoid overfitting
        lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
        # 2 layers lstm
        lstm_cell = rnn_cell.MultiRNNCell([lstm_cell]*2)   
        # Get lstm cell output
        outputs, states = rnn.rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.lstm_weights['out']) + self.lstm_biases['out'],outputs[-1]


    def build_arch(self):
        with tf.variable_scope('LSTM_layer'):
			#pred batch*4d out batch*128d 
            pred,out = self.RNN()
            out=tf.reshape(out,(-1,1,self.n_hidden,1))
			# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
			# #Adam optimizer
			# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
			# Evaluate model
			# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
			# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=self.kernel_size1, stride=1,
                                             padding='VALID')
#            print(conv1.get_shape(),[cfg.batch_size, self.conv1_outa,self.conv1_outb, 256])
            assert conv1.get_shape() == [cfg.batch_size, self.conv1_outa,self.conv1_outb, 256]
            
            out=tf.tile(out,[1,self.conv1_outa,1,256])
            self.conv1=tf.add(conv1,out)
#            out_temp= tf.placeholder(tf.float32, shape=(cfg.batch_size,self.conv1_outa+1,self.conv1_outb, 256))
#            self.dense1 = tf.layers.dense(inputs=tf.reshape(self.conv1,(cfg.batch_size,-1)), units=self.n_classes, activation=tf.nn.relu)
            #全连接层
            pool = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
            fc1 = tf.layers.dense(inputs=pool, units=1024, activation=tf.nn.relu)
            fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
            self.dense1 = tf.layers.dense(inputs=tf.reshape(fc2,(cfg.batch_size,-1)), units=self.n_classes, activation=None)
            self.dense1_index = tf.to_int32(tf.argmax(tf.nn.softmax(self.dense1, axis=1), axis=1))
            
            

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV', conv1_outa=self.conv1_outa, conv1_outb=self.conv1_outb, cap1_out=self.cap1_out,n_classes=self.n_classes)
            (self.caps1,pred) = primaryCaps(self.conv1, kernel_size=self.kernel_size2, stride=2)
            self.lstmpred=pred
            assert self.caps1.get_shape() == [cfg.batch_size, self.cap1_out, 8, 1]
            
#            self.dense2= tf.layers.dense(inputs=tf.reshape(self.caps1,(cfg.batch_size,-1)), units=self.n_classes, activation=tf.nn.relu)

            pool = tf.layers.max_pooling2d(inputs=self.caps1, pool_size=[2, 2], strides=2)
            fc1 = tf.layers.dense(inputs=pool, units=1024, activation=tf.nn.relu)
            fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
            self.dense2 = tf.layers.dense(inputs=tf.reshape(fc2,(cfg.batch_size,-1)), units=self.n_classes, activation=None)
            self.dense2_index = tf.to_int32(tf.argmax(tf.nn.softmax(self.dense2, axis=1), axis=1))



        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=self.n_classes, vec_len=8, with_routing=True, layer_type='FC',conv1_outa=self.conv1_outa, conv1_outb=self.conv1_outb, cap1_out=self.cap1_out,n_classes=self.n_classes)
            self.caps2 = digitCaps(self.caps1)
#            self.caps2 = tf.add(tf.tile(tf.reshape(self.lstmpred,(cfg.batch_size,self.n_classes,1,1)),[1,1,16,1]),self.caps2)

        
        
        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keepdims=True))
            self.softmax_v = tf.nn.softmax(self.v_length, axis=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, self.n_classes, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))
            
#            self.max_index_list= tf.Variable(tf.ones([cfg.batch_size, ],dtype=tf.int32))
#            index_list=tf.stack([self.dense1_index,self.dense2_index,self.argmax_idx],1)
#        
#            for i in range(cfg.batch_size):
#                max_index=tf.to_int32(tf.argmax(tf.bincount(index_list[i])))
#                self.update_op=tf.assign(self.max_index_list[i],max_index)



#            # Method 1.
#            if not cfg.mask_with_y:
#                # c). indexing
#                # It's not easy to understand the indexing process with argmax_idx
#                # as we are 3-dim animal
#                masked_v = []
#                for batch_size in range(cfg.batch_size):
#                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
#                    masked_v.append(tf.reshape(v, shape=(1, 1, 8, 1)))
#
#                self.masked_v = tf.concat(masked_v, axis=0)
#                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 8, 1]
#            # Method 2. masking with true label, default mode
#            else:
#                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
#                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, self.n_classes, 1)))
#                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)
#
#        # 2. Reconstructe the MNIST images with 3 FC layers
#        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
#        with tf.variable_scope('Decoder'):
#            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
#            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
#            assert fc1.get_shape() == [cfg.batch_size, 512]
#            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
#            assert fc2.get_shape() == [cfg.batch_size, 1024]
#            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=self.n_steps*self.n_input, activation_fn=tf.sigmoid)





    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size,self.n_classes, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

#        # 2. The reconstruction loss
#        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
#        squared = tf.square(self.decoded - orgin)
#        self.reconstruction_err = tf.reduce_mean(squared)
#        lstm loss
        self.lstm_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.lstmpred, labels=self.lstm_y))
        self.lstm_index = tf.to_int32(tf.argmax(self.lstmpred, axis=1))

        self.dense_cov1_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dense1, labels=self.lstm_y))
        self.dense_caps1_cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dense2, labels=self.lstm_y))
        
        
        
        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
#        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err+lstm_cost
#        self.total_loss = self.margin_loss+self.lstm_cost
        self.total_loss = self.margin_loss+self.lstm_cost+self.dense_cov1_cost+self.dense_caps1_cost


    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
#        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
#        train_summary.append(tf.summary.scalar('train/rf_loss', self.rf_loss_op))
#        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, self.n_input,self.n_steps, 1))
#        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)
#        correct_prediction = tf.equal(tf.to_int32(self.labels), self.max_index_list)
#        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))






