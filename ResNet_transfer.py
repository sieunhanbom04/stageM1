#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:46:59 2019

@author: quoctung
"""

import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from adversarialMMD import AdversarialNet
from ResNet_datagenerator import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input
import mutan_block as mb
from tensorflow.keras.applications.resnet50 import ResNet50
import time
import gc

Iterator = tf.data.Iterator
valmode = True # Use full taget set for evaluation
display_step = 15

proj_dim = 1024
config = [20, 20, 20, 20]
size = 224
height = 224
width = 224

source_file = './testtxt/amazon.txt'
target_file = './testtxt/webcam.txt'

class AdResNet(AdversarialNet):
    def __init__(self, source, target, beta = tf.placeholder(tf.float32), keep_prob = tf.placeholder(tf.float32), dimension = 256, lamda = tf.placeholder(tf.float32), skip_layer = ['fc8'], n_class = 31, weight_path = "bvlc_alexnet.npy"):
        self.source = source
        self.target = target
        self.n_source = tf.shape(source)[0]
        self.n_target = tf.shape(target)[0]
        self.beta = beta
        AlexNet.__init__(self, tf.concat([self.source, self.target], axis = 0), keep_prob, skip_layer, n_class, weight_path = "bvlc_alexnet.npy")
        self.rep_dim = dimension
        self.lamda = lamda
        self.KEEP_PROB_TRAINING = 0.5
        self.KEEP_PROB_VALIDATION = 1.0
        self.pretrain_weight = {}
        
    def create(self):
        self.model = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor = self.X)
        self.model.trainable = True
        for layer in self.model.layers:
            self.pretrain_weight[layer.name] = self.model.get_layer(layer.name).get_weights()
        print(self.model.get_layer('res5c_branch2a').get_weights())
        output_resnet = self.model(self.X)
            
        self.bottleneck = self.fc(output_resnet, 2048, self.rep_dim, name = "adapt/bottleneck")
        #self.dropout_bottleneck = self.dropout(self.bottleneck, self.keep_prob)
        self.logits = tf.layers.dense(self.bottleneck, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        
        self.source_classify, self.target_classifiy = tf.split(self.logits, [self.n_source, self.n_target])
        
        ad_input = tf.matmul(tf.expand_dims(self.bottleneck, axis = -1), tf.expand_dims(tf.nn.softmax(self.logits, axis = -1), axis = -1), transpose_b = True)
        ad_input = tf.reshape(ad_input, [-1, self.rep_dim * self.n_class])
        self.discriminator = self.adversarial_net(ad_input, [self.rep_dim * self.n_class, 1024], name = "feature")
                
        print(output_resnet)
    
    def create_block(self, version = 0, cdan = 0):
        self.model = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor = self.X)
        self.model.trainable = True
        for layer in self.model.layers:
            self.pretrain_weight[layer.name] = self.model.get_layer(layer.name).get_weights()
        output_resnet = self.model(self.X)
        
        adapt = self.fc(output_resnet, 2048, self.rep_dim * 2, name = "adapt/main_block1")
        adapt = self.fc(adapt, self.rep_dim * 2, self.rep_dim, name = "adapt/main_block2")
        adapt_source, adapt_target = tf.split(adapt, [self.n_source, self.n_target])
        if version == 0:
            fc7_skip = self.fc(output_resnet, 2048, self.rep_dim, name = "adapt/skip_connection")
        
            fc7_skip_source, fc7_skip_target = tf.split(fc7_skip, [self.n_source, self.n_target])
        
            self.source_feature = fc7_skip_source + self.beta * adapt_source
            self.target_feature = fc7_skip_target + adapt_target
        else:
            fc7_source, fc7_target = tf.split(output_resnet, [self.n_source, self.n_target])
            
            adapt_source = self.fc(fc7_source, 2048, self.rep_dim, name = "adapt/skip_connection_source")
            adapt_feature = self.fc(fc7_target, 2048, self.rep_dim, name = "adapt/skip_connection_target")
            
            self.source_feature = adapt_source + adapt_source
            self.target_feature = adapt_feature + adapt_target
        
        self.bottleneck = tf.concat([self.source_feature, self.target_feature], axis = 0) 
        self.dropout_bottleneck = self.dropout(self.bottleneck, self.keep_prob)
        self.logits = tf.layers.dense(self.dropout_bottleneck, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        
        self.source_classify, self.target_classifiy = tf.split(self.logits, [self.n_source, self.n_target])
        
        if cdan == 0:
            ad_input = tf.matmul(tf.expand_dims(self.bottleneck, axis = -1), tf.expand_dims(tf.nn.softmax(self.logits, axis = -1), axis = -1), transpose_b = True)
            ad_input = tf.reshape(ad_input, [-1, self.rep_dim * self.n_class])
            self.discriminator = self.adversarial_net(ad_input, [self.rep_dim * self.n_class, 1024], name = "feature")
        else:
            self.discriminator = self.mutan_adversarial_net(self.bottleneck, self.logits, [proj_dim, 1024], name = "feature")
        
        #self.source_logits, self.target_logits = tf.split(discriminator, [self.n_source, self.n_target])
        
        print(self.bottleneck)
    
    
    def adversarial_net(self, x, dimension, name = "feature"):
        x = self.dropout(x, self.keep_prob)
        x = self.fc(x, dimension[0], dimension[1], "adversarial_" + name + "_layer_2")
        x = self.dropout(x, self.keep_prob)
        x = tf.layers.dense(x, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name = "adversarial_" + name + "_layer_3")
        
        return x
    
    def mutan_adversarial_net(self, x, y, dimension, name = 'feature'):
        rep = mb.mutan_block(x, y, self.rep_dim, self.n_class, dimension[0], config)
        rep = self.adversarial_net(rep, dimension, name = name)
        return rep
    
    def create_mutan(self):
        self.model = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor = self.X)
        self.model.trainable = True
        for layer in self.model.layers:
            self.pretrain_weight[layer.name] = self.model.get_layer(layer.name).get_weights()
        print(self.model.get_layer('res5c_branch2a').get_weights())
        output_resnet = self.model(self.X)
            
        self.bottleneck = self.fc(output_resnet, 2048, self.rep_dim, name = "adapt/bottleneck")
        #self.dropout_bottleneck = self.dropout(self.bottleneck, self.keep_prob)
        self.logits = tf.layers.dense(self.bottleneck, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        
        self.source_classify, self.target_classifiy = tf.split(self.logits, [self.n_source, self.n_target])
        
        self.discriminator = self.mutan_adversarial_net(self.bottleneck, self.logits, [proj_dim, 1024], name = "feature")
    

    #init_lr = 1e-3, init_lr= 0.0003
    def training(self, source_file, target_file, init_lr = 5e-4, training_epochs = 1000, batch_size = 30, batch_test = 60):
        
        sess = tf.Session()
        K.set_session(sess)
        #self.create()
        #self.create_block()
        #self.create_mutan()
        self.create_block(version = 1, cdan = 0)
        
        """
        ----------------
        MODEL DEFINITION
        ----------------
        """
        one_hot_labels = tf.placeholder(tf.float32, [None, self.n_class])
        lr = tf.placeholder(tf.float32)
        domain_label = tf.concat([tf.ones([self.n_source, 1]), tf.zeros([self.n_target, 1])], axis = 0)
        
        non_adversarial = [v for v in tf.trainable_variables() if not ('adversarial' in v.name)]
        adversarial_list = [v for v in tf.trainable_variables() if 'adversarial' in v.name]
        scratch_list = [v for v in tf.trainable_variables() if 'adapt' in v.name]
        finetuning_list = [v for v in tf.trainable_variables() if not (('adapt' in v.name) or ('adversarial' in v.name))]
        
        weight_decay_not_ad = 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in non_adversarial if ('weight' in v.name) or ('kernel' in v.name)])
        weight_decay_ad = 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in adversarial_list if ('weight' in v.name) or ('kernel' in v.name)])
        #prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.source_classify, labels = one_hot_labels))
        prediction_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.source_classify, labels = tf.argmax(one_hot_labels, 1)))
        discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discriminator, labels = domain_label))
        adver_loss = discriminator_loss + weight_decay_ad
        
        skip_source_w = [v for v in tf.trainable_variables() if ('adapt/skip_connection_source/weights' in v.name)][0]
        skip_source_b = [v for v in tf.trainable_variables() if ('adapt/skip_connection_source/biases' in v.name)][0]
        skip_target_w = [v for v in tf.trainable_variables() if ('adapt/skip_connection_target/weights' in v.name)][0]
        skip_target_b = [v for v in tf.trainable_variables() if ('adapt/skip_connection_target/biases' in v.name)][0]
        
        print(skip_source_w.name, skip_source_b.name, skip_target_w.name, skip_target_b.name)
        difference_loss = (tf.reduce_sum(tf.square(skip_source_w - skip_target_w)) + tf.reduce_sum(tf.square(skip_source_b - skip_target_b)))
        loss = prediction_loss + weight_decay_not_ad - self.lamda * discriminator_loss + difference_loss
                
        """
        --------------------
        OPTIMIZER DEFINITION
        --------------------
        """
        
        #print(adversarial_list)
        #print(scratch_list)
        #print(finetuning_list)
        
        min_var_list = scratch_list + finetuning_list
        adv_var_list = adversarial_list
        
        min_gradients = tf.gradients(loss, min_var_list)
        adv_gradients = tf.gradients(adver_loss, adv_var_list)
        #min_gradients, _ = tf.clip_by_global_norm(min_gradients, 5.0)
        #adv_gradients, _ = tf.clip_by_global_norm(adv_gradients, 5.0)
        min_gradients = list(zip(min_gradients, min_var_list))
        adv_gradients = list(zip(adv_gradients, adv_var_list))
        
        optimizer1 = tf.train.MomentumOptimizer(lr * 10, momentum = 0.9)
        optimizer2 = tf.train.MomentumOptimizer(lr, momentum = 0.9)
        optimizer3 = tf.train.MomentumOptimizer(lr * 10, momentum = 0.9)
        #optimizer3 = tf.train.AdamOptimizer(lr * 10)        
        
        train_op1 = optimizer1.apply_gradients(grads_and_vars = min_gradients[:len(scratch_list)])
        train_op2 = optimizer2.apply_gradients(grads_and_vars = min_gradients[len(scratch_list):])
        train_op3 = optimizer3.apply_gradients(grads_and_vars = adv_gradients)
        train_op = tf.group(train_op1, train_op2, train_op3)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimize = tf.group([train_op, update_ops])
            
        """
        -----------------------------------------------
        DATA PREPARATION (REUSE FROM THE ALEXNET MODEL)
        -----------------------------------------------
        """
        
        """
        with tf.device('/cpu:0'):
            
            train_source_data = ImageDataGenerator(preprocessing_function = preprocess_input,
                                                   horizontal_flip = True,
                                                   rotation_range = 40,
                                                   shear_range=0.2)
            train_target_data = ImageDataGenerator(preprocessing_function = preprocess_input,
                                                   horizontal_flip = True,
                                                   rotation_range = 40,
                                                   shear_range=0.2)
            val_data = ImageDataGenerator(preprocessing_function = preprocess_input)
            
            train_source_generator = train_source_data.flow_from_directory(source_file,
                                                                           batch_size = batch_size,
                                                                           target_size = (size, size),
                                                                           class_mode = 'categorical',
                                                                           shuffle = True)
            
            train_target_generator = train_target_data.flow_from_directory(target_file,
                                                                           batch_size = batch_size,
                                                                           target_size = (size, size),
                                                                           class_mode = 'categorical',
                                                                           shuffle = True)
            
            val_generator = val_data.flow_from_directory(target_file,
                                                         batch_size = batch_test,
                                                         target_size = (size, size),
                                                         class_mode = 'categorical')
            
            
            
            
            #next_batch_sr = train_source_generator.next()
            #next_batch_tr = train_target_generator.next()
            #next_batch_val = val_generator.next() 
            
            train_batches_per_epoch = int(np.floor(train_source_generator.n/ train_source_generator.batch_size))
            val_batches_per_epoch = int(np.ceil(val_generator.n / val_generator.batch_size))
        """
        
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
        training_source_init_op = train_source_generator.make_initializer(train_source_data.data)
        training_target_init_op = train_target_generator.make_initializer(train_target_data.data)
        validation_init_op = val_generator.make_initializer(train_target_data.data)
        """
        
        """
        ---------------------------
        VALIDATION SCORE DEFINITION
        ---------------------------
        """
        prob = tf.placeholder(tf.float32, [None, self.n_class])
        correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
        accunum = tf.shape(one_hot_labels)[0]
        
        """
        ------------------------------
        ACTUAL TRAINING AND EVALUATION
        ------------------------------
        """
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        for layer in self.model.layers:
            self.model.get_layer(layer.name).set_weights(self.pretrain_weight[layer.name])
        #self.load_initial_weight(sess)
        del self.pretrain_weight
        gc.collect()
        
        print(init_lr)
        print(train_batches_per_epoch)
        print(val_batches_per_epoch)
        print(self.model.get_layer('res5c_branch2a').get_weights())
        print(source_file, target_file)
        #print(train_source_generator.class_indices)
        #print(train_target_generator.class_indices)
        #print(val_generator.class_indices)
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
            self.model.trainable = True
            
            for times in range(train_batches_per_epoch):
                if times_pass == val_batches_per_epoch - 1:
                    sess.run(training_target_init_op)
                    times_pass = 0
                else:
                    times_pass += 1
                    
                p = 0.5 * epochs / training_epochs # + times/train_batches_per_epoch/training_epochs
                learning_rate = init_lr / (1. + 10.0 * p)**0.75
                b = 1.0 * epochs / training_epochs
                #print(learning_rate)
                alpha = 0.25#2.0 / (1.0 + np.exp(-10.0 * epochs / training_epochs)) - 1.0#min(0.3, 2.0 / (1.0 + np.exp(-10.0 * epochs / training_epochs)) - 1.0) 
                #(1 - np.exp(-10.0 * epochs / training_epochs)) / (1 + np.exp(-10.0 * epochs / training_epochs)) 
                
                #begin = time.time()
                x1, y1 = sess.run(next_batch_sr)
                x2, y2 = sess.run(next_batch_tr)
                #end = time.time()
                #print("Loading: ", end - begin)
                
                #begin = time.time()
                #print(times, np.mean(x1), np.mean(x2))
                _, total_loss, pred_loss, M_loss, D_loss = sess.run([optimize, loss, prediction_loss, adver_loss, difference_loss], feed_dict = {self.source: x1, self.target : x2,
                                            one_hot_labels : y1, lr : learning_rate, self.keep_prob : self.KEEP_PROB_TRAINING, self.lamda : alpha, self.beta : b})
                #end = time.time()
                #print("Running: ",end - begin)
                
                #print(total_loss, pred_loss, M_loss)
                
                L_pred = L_pred + pred_loss
                L_total = L_total + total_loss
                L_MMD = L_MMD + M_loss
                L_diff = L_diff + D_loss
            
            
            gc.collect()
                
            if not (epochs % 10 == 9):
                continue
            
            self.model.trainable = False
            #print(sess.run([v for v in tf.trainable_variables() if 'conv1/weights' in v.name]))
            print(L_pred * 1.0 / train_batches_per_epoch)
            print(L_MMD / train_batches_per_epoch)
            print(L_diff * 1.0 / train_batches_per_epoch)
            print(alpha)
            print("TIMES: ", epochs, " LOSS: ", L_total * 1.0 / train_batches_per_epoch)
            
            sess.run(validation_init_op)
            
            test_acc = 0.
            num = 0.
            
            for _ in range(val_batches_per_epoch):
                val_input = 0
                img_batch, label_batch = sess.run(next_batch_val)
                #img_batch = self.crop_generator(img_batch, False)
                #img_batch = np.reshape(img_batch, [-1 ,height, width, 3])
                #print(img_batch.size)
                logits = sess.run([tf.nn.softmax(self.logits, axis = -1)], feed_dict={self.target: img_batch, self.source : np.zeros(shape = (0,height, width,3), dtype = np.float32), 
                                       self.keep_prob : self.KEEP_PROB_VALIDATION, self.beta : 1.0})
                                    
                val_input += logits[0]
                
                acc, num_ = sess.run([accuracy, accunum], feed_dict={prob: val_input, one_hot_labels: label_batch})
                test_acc += acc
                num += num_
            
            print("TIMES: ", epochs, " VALIDATION: ", test_acc / num)
            best = max(best, test_acc / num)
            
        print (best)
        
if source_file == './testtxt/dslr.txt':
    training_epochs = 400
elif source_file == './testtxt/webcam.txt' :
    training_epochs = 400
else:
    training_epochs = 400 

net = AdResNet(tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]), tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]))
net.training(source_file, target_file, training_epochs = training_epochs)