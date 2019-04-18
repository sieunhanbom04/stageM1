#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:22:02 2019

@author: quoctung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:19:01 2019

@author: quoctung
"""

import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
import datagenerator


Iterator = tf.data.Iterator
valmode = True # Use full taget set for evaluation
display_step = 15

height = datagenerator.image_height
width = datagenerator.image_width

source_file = './testtxt/amazon.txt'
target_file = './testtxt/dslr.txt'
size = len(open(source_file).readlines())

class AdversarialNet(AlexNet):
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
        self.create()
        
    def create(self):
        super().create()
            
        self.bottleneck = self.fc(self.dropout7, 4096, self.rep_dim, name = "adapt/bottleneck")
        self.dropout_bottleneck = self.dropout(self.bottleneck, self.keep_prob)
        self.logits = tf.layers.dense(self.dropout_bottleneck, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        
        self.source_classify, self.target_classifiy = tf.split(self.logits, [self.n_source, self.n_target])
        
        ad_input = tf.matmul(tf.expand_dims(self.bottleneck, axis = -1), tf.expand_dims(tf.nn.softmax(self.logits, axis = -1), axis = -1), transpose_b = True)
        ad_input = tf.reshape(ad_input, [-1, self.rep_dim * self.n_class])
        discriminator = self.adversarial_net(ad_input, [self.rep_dim * self.n_class, 1024], name = "feature")
        
        self.source_logits, self.target_logits = tf.split(discriminator, [self.n_source, self.n_target])
        
        print(self.bottleneck)
        
    def adversarial_net(self, x, dimension, name = "feature"):
        x = self.fc(x, dimension[0], dimension[1], "adversarial_" + name + "_layer_1")
        #x = self.dropout(x, self.keep_prob)
        x = self.fc(x, dimension[1], dimension[1], "adversarial_" + name + "_layer_2")
        #x = self.dropout(x, self.keep_prob)
        x = tf.layers.dense(x, 1, activation = tf.nn.sigmoid, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name = "adversarial_" + name + "_layer_3")
        
        return x
    
    def MMD_loss(self, source, target, batch_size = 100):
        feature = tf.concat([source[0], target[0]], axis = 0)
        logits = tf.concat([source[1], target[1]], axis = 0)
        feature_kernel = tf.matmul(feature, feature, transpose_b = True)
        logits_kernel = tf.matmul(logits, logits, transpose_b = True)
        
        joint_kernel = feature_kernel * logits_kernel
        
        
        XX = joint_kernel[:self.n_source, :self.n_source]
        YY = joint_kernel[self.n_source:, self.n_source:]
        XY = joint_kernel[:self.n_source, self.n_source:]
        YX = joint_kernel[self.n_source:, :self.n_source]
        
        return tf.reduce_mean(XX + YY - XY - YX)
        
        
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
        
    #init_lr = 1e-3, init_lr= 0.0003
    def training(self, source_file, target_file, init_lr = 1e-3, training_epochs = 300, batch_size = 100, batch_test = 30):
        
        sess = tf.Session()
        """
        ----------------
        MODEL DEFINITION
        ----------------
        """
        one_hot_labels = tf.placeholder(tf.float32, [None, self.n_class])
        lr = tf.placeholder(tf.float32)
        
        weight_decay = 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name) or ('kernel' in v.name)])
        prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.source_classify, labels = one_hot_labels))
        adver_loss = - (tf.reduce_mean(tf.log(self.source_logits)) + tf.reduce_mean(tf.log(1 - self.target_logits)))
        loss = prediction_loss + weight_decay - self.lamda * adver_loss
                
        """
        --------------------
        OPTIMIZER DEFINITION
        --------------------
        """
        
        adversarial_list = [v for v in tf.trainable_variables() if 'adversarial' in v.name]
        scratch_list = [v for v in tf.trainable_variables() if 'adapt' in v.name]
        finetuning_list = [v for v in tf.trainable_variables() if not (('adapt' in v.name) or ('adversarial' in v.name))]
        
        #print(adversarial_list)
        #print(scratch_list)
        #print(finetuning_list)
        min_var_list = scratch_list + finetuning_list
        adv_var_list = adversarial_list
        
        min_gradients = tf.gradients(loss, min_var_list)
        adv_gradients = tf.gradients(adver_loss, adv_var_list)
        min_gradients = list(zip(min_gradients, min_var_list))
        adv_gradients = list(zip(adv_gradients, adv_var_list))
        
        optimizer1 = tf.train.MomentumOptimizer(lr * 10, momentum = 0.9)
        optimizer2 = tf.train.MomentumOptimizer(lr, momentum = 0.9)
        optimizer3 = tf.train.MomentumOptimizer(lr * 10, momentum = 0.9)
                
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
                                      batch_size=batch_test,
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
        prob = tf.nn.softmax(self.logits, axis = -1)
        prob = tf.split(prob, batch_test)
        prob = tf.concat([tf.reduce_mean(i) for i in prob], axis = 0)
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
                
        self.load_initial_weight(sess)
        
        print(train_batches_per_epoch)
        print(val_batches_per_epoch)
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
            
            for times in range(train_batches_per_epoch):
                if times_pass == val_batches_per_epoch - 1:
                    sess.run(training_target_init_op)
                    times_pass = 0
                else:
                    times_pass += 1
                    
                p = 0.5 * epochs / training_epochs # + times/train_batches_per_epoch/training_epochs
                learning_rate = init_lr / (1. + 10.0 * p)**0.75
                #print(learning_rate)
                alpha = 2.0 / (1.0 + np.exp(-10.0 * epochs / training_epochs)) - 1
                #(1 - np.exp(-10.0 * epochs / training_epochs)) / (1 + np.exp(-10.0 * epochs / training_epochs)) 
                
                x1, y1 = sess.run(next_batch_sr)
                x2, y2 = sess.run(next_batch_tr)                
                
                #print(times, np.mean(x1), np.mean(x2))
                _, total_loss, pred_loss, M_loss = sess.run([optimize, loss, prediction_loss, adver_loss], feed_dict = {self.source: x1, self.target : x2,
                                            one_hot_labels : y1, lr : learning_rate, self.keep_prob : self.KEEP_PROB_TRAINING, self.lamda : alpha})
                
                L_pred = L_pred + pred_loss
                L_total = L_total + total_loss
                L_MMD = L_MMD + M_loss
    
            if not (epochs % 5 == 4):
                continue
            
            #print(sess.run([v for v in tf.trainable_variables() if 'conv1/weights' in v.name]))
            print(L_pred * 1.0 / train_batches_per_epoch)
            print(L_MMD / train_batches_per_epoch)
            print("TIMES: ", epochs, " LOSS: ", L_total * 1.0 / train_batches_per_epoch)
            
            sess.run(validation_init_op)
            
            test_acc = 0.
            num = 0.
            for _ in range(val_batches_per_epoch):
                
                img_batch, label_batch = sess.run(next_batch_val)
                #img_batch = np.reshape(img_batch, [-1 ,height, width, 3])
                #print(img_batch.size)
                acc, num_ = sess.run([accuracy, accunum], feed_dict={self.source: img_batch, self.target : np.zeros(shape = (0,height, width,3), dtype = np.float32), 
                                                one_hot_labels: label_batch, self.keep_prob : self.KEEP_PROB_VALIDATION})
                    
                test_acc += acc
                num += num_
            
            print("TIMES: ", epochs, " VALIDATION: ", test_acc / num)
            best = max(best, test_acc / num)
            
        print (best)

net = AdversarialNet(tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]), tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]))
net.training(source_file, target_file)