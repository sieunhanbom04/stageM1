#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:19:01 2019

@author: quoctung
"""

import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datagenerator import ImageDataGenerator
import datagenerator


Iterator = tf.data.Iterator
valmode = True # Use full taget set for evaluation
display_step = 15

height = datagenerator.image_height
width = datagenerator.image_width

source_file = './testtxt/dslr.txt'
target_file = './testtxt/webcam.txt'
size = len(open(source_file).readlines())

class AdaptNet(AlexNet):
    def __init__(self, source, target, beta = tf.placeholder(tf.float32), mode = "JAN", keep_prob = tf.placeholder(tf.float32), dimension = 256, lamda = tf.placeholder(tf.float32), skip_layer = ['fc8'], n_class = 31, weight_path = "bvlc_alexnet.npy"):
        self.source = source
        self.target = target
        self.n_source = tf.shape(source)[0]
        self.n_target = tf.shape(target)[0]
        self.beta = beta
        AlexNet.__init__(self, tf.concat([self.source, self.target], axis = 0), keep_prob, skip_layer, n_class, weight_path = "bvlc_alexnet.npy")
        self.rep_dim = dimension
        self.lamda = lamda
        self.mode = mode
        self.KEEP_PROB_TRAINING = 0.5
        self.KEEP_PROB_VALIDATION = 1.0
        #self.create()
        #self.create_block_ver2()
        self.create_block_ver3()
        
    def create(self):
        super().create()
        
        if 'adaptation_block' in self.mode:
        #The adaptation block
            fc7_skip = self.fc(self.dropout7, 4096, self.rep_dim, name = "adapt/skip_connection")
        
            adapt = self.fc(self.dropout7, 4096, self.rep_dim * 2, name = "adapt/main_block1")
            adapt = self.fc(adapt, self.rep_dim * 2, self.rep_dim, name = "adapt/main_block2")
        
            fc7_skip_source, fc7_skip_target = tf.split(fc7_skip, [self.n_source, self.n_target])
            adapt_source, adapt_target = tf.split(adapt, [self.n_source, self.n_target])
        
            self.source_feature = fc7_skip_source + self.beta * adapt_source
            self.target_feature = fc7_skip_target + adapt_target
        
            self.fc8 = tf.concat([self.source_feature, self.target_feature], axis = 0) 
        
            #Feature and logits extraction
            self.dropout8 = self.dropout(self.fc8, self.keep_prob)
            self.logits = tf.layers.dense(self.dropout8, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
                
            print(self.logits, self.fc8)
            self.skip_layer = ['fc8', 'adapt']
            
        elif self.mode == 'JAN':
            self.bottleneck = self.fc(self.dropout7, 4096, self.rep_dim, name = "adapt/feature_map")
            self.source_feature, self.target_feature = tf.split(self.bottleneck, [self.n_source, self.n_target]) 
            self.dropout_bottleneck = self.dropout(self.bottleneck, self.keep_prob)
            self.logits = tf.layers.dense(self.dropout_bottleneck, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
            
            print(self.bottleneck)
    #Gaussian Kernel
    
    def create_block_ver2(self):
        super().create()
        
        adapt = self.fc(self.dropout7, 4096, self.rep_dim * 2, name = "adapt/main_block1")
        adapt = self.fc(adapt, self.rep_dim * 2, self.rep_dim, name = "adapt/main_block2")
        adapt_source, adapt_target = tf.split(adapt, [self.n_source, self.n_target])
        fc7_source, fc7_target = tf.split(self.dropout7, [self.n_source, self.n_target])
            
        adapt_source = self.fc(fc7_source, 4096, self.rep_dim, name = "adapt/skip_connection_source")
        adapt_feature = self.fc(fc7_target, 4096, self.rep_dim, name = "adapt/skip_connection_target")
            
        self.source_feature = adapt_source + adapt_source
        self.target_feature = adapt_feature + adapt_target
        
        self.fc8 = tf.concat([self.source_feature, self.target_feature], axis = 0) 
        
        #Feature and logits extraction
        self.dropout8 = self.dropout(self.fc8, self.keep_prob)
        self.logits = tf.layers.dense(self.dropout8, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        self.skip_layer = ['fc8', 'adapt']
        
    def create_block_ver3(self):
        super().create()
        
        adapt = self.fc(self.dropout7, 4096, self.rep_dim * 2, name = "adapt/main_block1")
        adapt = self.fc(adapt, self.rep_dim * 2, self.rep_dim, name = "adapt/main_block2")
        adapt_source, adapt_target = tf.split(adapt, [self.n_source, self.n_target])
        
        self.source_feature = adapt_source
        self.target_feature = adapt_target + self.fc(adapt_target, self.rep_dim, self.rep_dim, name = "adapt/projection")
        
        self.fc8 = tf.concat([self.source_feature, self.target_feature], axis = 0) 
        
        #Feature and logits extraction
        self.dropout8 = self.dropout(self.fc8, self.keep_prob)
        self.logits = tf.layers.dense(self.dropout8, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        self.skip_layer = ['fc8', 'adapt']
    
    def gaussian_kernel(self, source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma = None):
        total = tf.concat([source, target], axis = 0)
        
        n_source = tf.shape(source)[0]
        n_target = tf.shape(target)[0]
        n_sample = n_source + n_target
        
        r = tf.reduce_sum(tf.square(total), 1)
        
        r = tf.reshape(r, [-1,1])
        L2_distance = r - 2 * tf.matmul(total, tf.transpose(total)) + tf.transpose(r)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum(L2_distance) / tf.cast(n_sample * n_sample - n_sample, dtype = tf.float32) 
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        weight = 0
        for i in range(kernel_num):
            weight += tf.exp(-L2_distance / (bandwidth * (kernel_mul ** i)))
        
        return weight
    
    def laplace_kernel(self, source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma = None):
        total = tf.concat([source, target], axis = 0)
        
        n_source = tf.shape(source)[0]
        n_target = tf.shape(target)[0]
        n_sample = n_source + n_target
    
        L1_distance = tf.reduce_sum(tf.abs(tf.subtract(total, tf.expand_dims(total, 1))), axis = 2)
    
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum(L1_distance) / tf.cast(n_sample * n_sample - n_sample, dtype = tf.float32) 
    
        bandwidth /= kernel_mul ** (kernel_num // 2)
        weight = 0
        for i in range(kernel_num):
            weight += tf.exp(-L1_distance / (bandwidth * (kernel_mul ** i)))
    
        return weight
    
    #Deep Adaptation Network Maximum Mean Discrepancy
    
    def DAN_MMD(self, source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma = None):
        weight = self.gaussian_kernel(source, target)
        
        XX = weight[:self.n_source, :self.n_source]
        YY = weight[self.n_source:, self.n_source:]
        XY = weight[:self.n_source, self.n_source:]
        YX = weight[self.n_source:, :self.n_source]
        
        result = tf.reduce_mean(XX + YY - XY - YX)
        
        return result
    
    #Joint Adaptation Network Maximum Mean Discrepancy
    
    def JAN_MMD(self, source, target, batch_size = 100, kernel_mul = [2.0, 2.0], kernel_nums = [5, 1], fix_sigma = [None, 1.68]):
        joint_kernel = 0
        flag = False
        
        for i in range(len(source)):
            if flag:
                joint_kernel = joint_kernel * self.gaussian_kernel(source[i], target[i], kernel_mul[i], kernel_nums[i], fix_sigma[i]) 
            else:
                joint_kernel = self.gaussian_kernel(source[i], target[i], kernel_mul[i], kernel_nums[i], fix_sigma[i]) 
                flag = True
        
        
        #Full JAN
        XX = joint_kernel[:self.n_source, :self.n_source]
        YY = joint_kernel[self.n_source:, self.n_source:]
        XY = joint_kernel[:self.n_source, self.n_source:]
        YX = joint_kernel[self.n_source:, :self.n_source]
        
        result = tf.reduce_mean(XX + YY - XY - YX)
        
        return result
        
        
        """
        #accelerating JAN
        loss = 0
        
        
        print("Joint kernel: ", joint_kernel)
        for i in range(batch_size):
            s1, s2 = i, (i+1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += joint_kernel[s1, s2] + joint_kernel[t1, t2]
            loss -= joint_kernel[s1, t2] + joint_kernel[s2, t1]             
                    
        return loss / float(batch_size)
        """
    
    #kernel_nums = [5,5], fix_sigma = [None, None], kernel_nums = [5, 1], fix_sigma = [None, 1.68]
    def JLGAN_MMD(self, source, target, batch_size = 100, kernel_mul = [2.0, 2.0], kernel_nums = [5,5], fix_sigma = [None, None]):
        joint_kernel = 0
        flag = False
        
        for i in range(len(source)):
            if flag:
                joint_kernel = joint_kernel * self.laplace_kernel(source[i], target[i], kernel_mul[i], kernel_nums[i], fix_sigma[i]) 
            else:
                joint_kernel = self.laplace_kernel(source[i], target[i], kernel_mul[i], kernel_nums[i], fix_sigma[i]) 
                flag = True
         
        #Full JAN
        XX = joint_kernel[:self.n_source, :self.n_source]
        YY = joint_kernel[self.n_source:, self.n_source:]
        XY = joint_kernel[:self.n_source, self.n_source:]
        YX = joint_kernel[self.n_source:, :self.n_source]
        
        result = tf.reduce_mean(XX + YY - XY - YX)
        
        return result
    
    def MJAN_MMD(self, x_source, x_target, y_source, y_target,  batch_size = 100, kernel_mul = [2.0, 2.0], kernel_nums = [5, 1], fix_sigma = [None, 1.68]):
        c = np.exp(- 2 / fix_sigma[1])
        print(c)
        prob_total = tf.concat([y_source, y_target], axis = 0)
        feature_kernel = self.gaussian_kernel(x_source, x_target, kernel_mul[0], kernel_nums[0], fix_sigma[0])
        probability_kernel = tf.matmul(prob_total, prob_total, transpose_b = True)
        joint_kernel = c * feature_kernel + (1 - c) * feature_kernel * probability_kernel
        
        XX = joint_kernel[:self.n_source, :self.n_source]
        YY = joint_kernel[self.n_source:, self.n_source:]
        XY = joint_kernel[:self.n_source, self.n_source:]
        YX = joint_kernel[self.n_source:, :self.n_source]
        
        result = tf.reduce_mean(XX + YY - XY - YX)
        
        return result
    
    def fine_tuning_training(self, source_file, target_file, init_lr = 1e-3, training_epochs = 200, batch_size = 100):
        
        sess = tf.Session()
        """
        ----------------
        MODEL DEFINITION
        ----------------
        """
        one_hot_labels = tf.placeholder(tf.float32, [None, self.n_class])
        lr = tf.placeholder(tf.float32)
        
        if self.mode == 'fine_tuning' or self.mode == 'vgg_fine_tuning':
            classifier = self.logits
            prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = classifier, labels = one_hot_labels))
            loss = prediction_loss + 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name) or ('kernel' in v.name)])
        elif self.mode == 'adaptation_block':
            source_logits, target_logits = tf.split(self.logits, [self.n_source, self.n_target])
            prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = source_logits, labels = one_hot_labels))
            MMD_loss = self.DAN_MMD(self.source_feature, self.target_feature)
            loss = prediction_loss + 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name) or ('kernel' in v.name)]) + self.lamda * MMD_loss
        elif 'JAN' in self.mode:
            source_logits, target_logits = tf.split(self.logits, [self.n_source, self.n_target])
            prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = source_logits, labels = one_hot_labels))
            MMD_loss = self.JAN_MMD([self.source_feature, tf.nn.softmax(source_logits, axis = -1)], [self.target_feature, tf.nn.softmax(target_logits, axis = -1)])
            #MMD_loss = 2 * self.MJAN_MMD(self.source_feature, self.target_feature, tf.nn.softmax(source_logits, axis = -1), tf.nn.softmax(target_logits, axis = -1))
            """
            skip_source_w = [v for v in tf.trainable_variables() if ('adapt/skip_connection_source/weights' in v.name)][0]
            skip_source_b = [v for v in tf.trainable_variables() if ('adapt/skip_connection_source/biases' in v.name)][0]
            skip_target_w = [v for v in tf.trainable_variables() if ('adapt/skip_connection_target/weights' in v.name)][0]
            skip_target_b = [v for v in tf.trainable_variables() if ('adapt/skip_connection_target/biases' in v.name)][0]
            
            difference_loss = (tf.reduce_sum(tf.square(skip_source_w - skip_target_w)) + tf.reduce_sum(tf.square(skip_source_b - skip_target_b)))
            """
            
            difference_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'projection' in v.name])
            loss = prediction_loss + 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name) or ('kernel' in v.name)]) + self.lamda * MMD_loss + difference_loss
        
        print("check:", [v for v in tf.trainable_variables() if ('weight' in v.name) or ('kernel' in v.name)])
        """
        --------------------
        OPTIMIZER DEFINITION
        --------------------
        """
        
        if self.mode == 'fine_tuning':
            scratch_list = [v for v in tf.trainable_variables() if 'logits' in v.name]
            finetuning_list = [v for v in tf.trainable_variables() if not ('fc8' in v.name)]
        elif self.mode == 'vgg_fine_tuning':
            scratch_list = [v for v in tf.trainable_variables() if 'fc8' in v.name]
            finetuning_list = [v for v in tf.trainable_variables() if not ('fc8' in v.name)]
        elif 'adaptation_block' in self.mode:
            scratch_list = [v for v in tf.trainable_variables() if 'adapt' in v.name]
            finetuning_list = [v for v in tf.trainable_variables() if not ('adapt' in v.name)]
        elif self.mode == 'JAN':
            scratch_list = [v for v in tf.trainable_variables() if 'adapt' in v.name]
            finetuning_list = [v for v in tf.trainable_variables() if not ('adapt' in v.name)]
        
        print(scratch_list)
        print(finetuning_list)
        var_list = scratch_list + finetuning_list
        
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
        
        optimizer1 = tf.train.MomentumOptimizer(lr * 10, momentum = 0.9)
        optimizer2 = tf.train.MomentumOptimizer(lr, momentum = 0.9)
                
        train_op1 = optimizer1.apply_gradients(grads_and_vars = gradients[:len(scratch_list)])
        train_op2 = optimizer2.apply_gradients(grads_and_vars = gradients[len(scratch_list):])
        train_op = tf.group(train_op1, train_op2)
        
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimize = tf.group([train_op, update_ops])
            
            
        """
        -----------------------------------------------
        DATA PREPARATION (REUSE FROM THE ALEXNET MODEL)
        -----------------------------------------------
        """
        
        size = len(open(source_file).readlines())
        valmode = True
        
        with tf.device('/cpu:0'):
            
            train_source_data = ImageDataGenerator(source_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=self.n_class,
                                     shuffle=True, size=size)
            
            train_target_data = ImageDataGenerator(target_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=self.n_class,
                                     shuffle=True, size=size)
            
            val_data = ImageDataGenerator(target_file,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=self.n_class,
                                      shuffle=False, fulval=valmode)
            
            iterator_source_train = Iterator.from_structure(train_source_data.data.output_types,
                                       train_source_data.data.output_shapes)
            iterator_target_train = Iterator.from_structure(train_target_data.data.output_types,
                                       train_target_data.data.output_shapes)
            iterator_val = Iterator.from_structure(val_data.data.output_types,
                                       val_data.data.output_shapes)
            
            next_batch_sr = iterator_source_train.get_next()
            next_batch_tr = iterator_target_train.get_next()
            next_batch_val = iterator_val.get_next()
            
        train_batches_per_epoch = int(np.floor(train_source_data.data_size/batch_size))
        val_batches_per_epoch = int(np.ceil(val_data.data_size / batch_size))
        
        training_source_init_op = iterator_source_train.make_initializer(train_source_data.data)
        training_target_init_op = iterator_target_train.make_initializer(train_target_data.data)
        validation_init_op = iterator_val.make_initializer(val_data.data)
        
        """
        ---------------------------
        VALIDATION SCORE DEFINITION
        ---------------------------
        """
        print(self.logits)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
        accunum = tf.shape(one_hot_labels)[0]
        
        """
        ------------------------------
        ACTUAL TRAINING AND EVALUATION
        ------------------------------
        """
        
        init = tf.global_variables_initializer()
        sess.run(init)
                
        self.load_initial_weight(sess)
        
        print(train_batches_per_epoch)
        print(val_batches_per_epoch)
        print(self.mode)
        print(source_file, target_file)
        
        best = 0
        
        #print(sess.run([v for v in tf.trainable_variables() if 'conv1/weights' in v.name]))
        for epochs in range(training_epochs):
            
            sess.run(training_source_init_op)
            sess.run(training_target_init_op)
            times_pass = 0
            #print(sess.run([tf.get_default_graph().get_tensor_by_name('conv1/weights:0')]))
            L_pred = 0
            L_total = 0
            L_MMD = 0
            L_diff = 0
            
            for times in range(train_batches_per_epoch):
                if times_pass == val_batches_per_epoch - 1:
                    sess.run(training_target_init_op)
                    times_pass = 0
                else:
                    times_pass += 1
                    
                p = 0.5 * epochs / training_epochs # + times/train_batches_per_epoch/training_epochs
                learning_rate = init_lr / (1. + 10.0 * p)**0.75
                #print(learning_rate)
                alpha = 0.5#2.0 / (1 + np.exp(-10.0 * epochs / training_epochs)) - 1 #min(0.45, 2.0 / (1 + np.exp(-10.0 * epochs / training_epochs)) - 1)
                b = 1.0 * epochs / training_epochs
                
                x1, y1 = sess.run(next_batch_sr)
                x2, y2 = sess.run(next_batch_tr)                
                
                #print(times, np.mean(x1), np.mean(x2))
                if self.mode == 'fine_tuning':
                    _ , total_loss, pred_loss = sess.run([optimize, loss, prediction_loss], feed_dict = {self.source : x1, self.target : np.zeros(shape = (0, height, width, 3), dtype = np.float32), 
                                            one_hot_labels : y1, lr : learning_rate})
                elif self.mode == 'vgg_fine_tuning':
                    _, total_loss, pred_loss = sess.run([optimize, loss, prediction_loss], feed_dict = {self.source: x1, self.target : np.zeros(shape = (0, height, width, 3), dtype = np.float32),
                                            one_hot_labels : y1, lr : learning_rate, self.keep_prob : self.KEEP_PROB_TRAINING})
                elif 'adaptation_block' in self.mode:
                    _, total_loss, pred_loss = sess.run([optimize, loss, prediction_loss], feed_dict = {self.source: x1, self.target : x2,
                                            one_hot_labels : y1, lr : learning_rate, self.keep_prob : self.KEEP_PROB_TRAINING, self.lamda : alpha, self.beta : b})
                elif self.mode == 'JAN':
                     _, total_loss, pred_loss, M_loss, diff_loss = sess.run([optimize, loss, prediction_loss, MMD_loss, difference_loss], feed_dict = {self.source: x1, self.target : x2,
                                            one_hot_labels : y1, lr : learning_rate, self.keep_prob : self.KEEP_PROB_TRAINING, self.lamda : alpha})
                
                L_pred = L_pred + pred_loss
                L_total = L_total + total_loss
                L_MMD = L_MMD + M_loss
                L_diff = L_diff + diff_loss
                
    
            if not (epochs % 10 == 9):
                continue
            
            #print(sess.run([v for v in tf.trainable_variables() if 'conv1/weights' in v.name]))
            print(L_pred * 1.0 / train_batches_per_epoch)
            print(L_MMD / train_batches_per_epoch)
            print(L_diff / train_batches_per_epoch)
            print(alpha)
            print("TIMES: ", epochs, " LOSS: ", L_total * 1.0 / train_batches_per_epoch)
            
            sess.run(validation_init_op)
            
            test_acc = 0.
            num = 0.
            for _ in range(val_batches_per_epoch):
                
                img_batch, label_batch = sess.run(next_batch_val)
                #print(np.mean(img_batch))
                if self.mode == 'fine_tuning':
                    acc, num_ = sess.run([accuracy, accunum], feed_dict={self.source: img_batch, self.target : np.zeros(shape = (0,height, width,3), dtype = np.float32), 
                                                one_hot_labels: label_batch})
                elif 'adaptation_block' in self.mode:
                    acc, num_ = sess.run([accuracy, accunum], feed_dict={self.source: img_batch, self.target : np.zeros(shape = (0,height, width,3), dtype = np.float32), 
                                                one_hot_labels: label_batch, self.keep_prob : self.KEEP_PROB_VALIDATION, self.beta : 1.0})
                elif self.mode == 'JAN':
                    acc, num_ = sess.run([accuracy, accunum], feed_dict={self.target: img_batch, self.source : np.zeros(shape = (0,height, width,3), dtype = np.float32), 
                                                one_hot_labels: label_batch, self.keep_prob : self.KEEP_PROB_VALIDATION})
                    
                test_acc += acc
                num += num_
            
            print("TIMES: ", epochs, " VALIDATION: ", test_acc / num)
            best = max(best, test_acc / num)
            
        print (best)

if source_file == './testtxt/dslr.txt':
    training_epochs = 700
elif source_file == './testtxt/webcam.txt' :
    training_epochs = 600
else:
    training_epochs = 400 

net = AdaptNet(tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]), tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]))
net.fine_tuning_training(source_file, target_file, training_epochs = training_epochs)