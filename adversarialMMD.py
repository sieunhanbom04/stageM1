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
import mutan_block as mb

Iterator = tf.data.Iterator
valmode = True # Use full taget set for evaluation
display_step = 15

proj_dim = 1024
config = [20, 20, 20, 20]
height = datagenerator.image_height
width = datagenerator.image_width

source_file = './testtxt/dslr.txt'
target_file = './testtxt/amazon.txt'
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
        #self.create()
        #self.create_block()
        self.create_mutan()
        
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
    
    def create_block(self):
        super().create()
        
        fc7_skip = self.fc(self.dropout7, 4096, self.rep_dim, name = "adapt/skip_connection")
        
        adapt = self.fc(self.dropout7, 4096, self.rep_dim * 2, name = "adapt/main_block1")
        adapt = self.fc(adapt, self.rep_dim * 2, self.rep_dim, name = "adapt/main_block2")
        
        fc7_skip_source, fc7_skip_target = tf.split(fc7_skip, [self.n_source, self.n_target])
        adapt_source, adapt_target = tf.split(adapt, [self.n_source, self.n_target])
        
        self.source_feature = fc7_skip_source + self.beta * adapt_source
        self.target_feature = fc7_skip_target + adapt_target
        
        self.bottleneck = tf.concat([self.source_feature, self.target_feature], axis = 0) 
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
        if source_file == './testtxt/amazon.txt':
            x = self.dropout(x, self.keep_prob)
        x = self.fc(x, dimension[1], dimension[1], "adversarial_" + name + "_layer_2")
        if source_file == './testtxt/amazon.txt':
            x = self.dropout(x, self.keep_prob)
        x = tf.layers.dense(x, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name = "adversarial_" + name + "_layer_3")
        
        return x
    
    def create_mutan(self):
        super().create()
            
        self.bottleneck = self.fc(self.dropout7, 4096, self.rep_dim, name = "adapt/bottleneck")
        self.dropout_bottleneck = self.dropout(self.bottleneck, self.keep_prob)
        self.logits = tf.layers.dense(self.dropout_bottleneck, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="adapt/fc8")
        
        self.source_classify, self.target_classifiy = tf.split(self.logits, [self.n_source, self.n_target])

        self.discriminator = self.mutan_adversarial_net(self.bottleneck, self.logits, [proj_dim, 1024], name = "feature")
        
        #self.source_logits, self.target_logits = tf.split(discriminator, [self.n_source, self.n_target])
    
    def mutan_adversarial_net(self, x, y, dimension, name = 'feature'):
        rep = mb.mutan_block(x, y, self.rep_dim, self.n_class, dimension[0], config)
        rep = self.adversarial_net(rep, dimension, name = name)
        return rep
        #return tf.nn.l2_normalize(tf.nn.relu(rep) - tf.nn.relu(-rep), axis = -1)
        
    #init_lr = 1e-3, init_lr= 0.0003
    def training(self, source_file, target_file, init_lr = 5e-4, training_epochs = 1000, batch_size = 100, batch_test = 100):
        
        sess = tf.Session()
        """
        ----------------
        MODEL DEFINITION
        ----------------
        """
        one_hot_labels = tf.placeholder(tf.float32, [None, self.n_class])
        lr = tf.placeholder(tf.float32)
        domain_label = tf.concat([tf.ones([self.n_source, 1]), tf.zeros([self.n_target, 1])], axis = 0)
        
        weight_decay = 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name) or ('kernel' in v.name)])
        #prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.source_classify, labels = one_hot_labels))
        prediction_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.source_classify, labels = tf.argmax(one_hot_labels, 1)))
        adver_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discriminator, labels = domain_label))
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
        #min_gradients, _ = tf.clip_by_global_norm(min_gradients, 5.0)
        #adv_gradients, _ = tf.clip_by_global_norm(adv_gradients, 5.0)
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
            
            val_data = []
            for t in range(9):
                i = t // 3
                j = t % 3
                val_data.append(ImageDataGenerator(target_file,
                                      mode='inference',
                                      batch_size=batch_test,
                                      num_classes=self.n_class,
                                      shuffle=False, fulval=valmode, pos_x = i, pos_y = j))
            
            iterator_source_train = Iterator.from_structure(train_source_data.data.output_types,
                                       train_source_data.data.output_shapes)
            iterator_target_train = Iterator.from_structure(train_target_data.data.output_types,
                                       train_target_data.data.output_shapes)
            
            iterator_val = []
            for t in range(9):
                iterator_val.append(Iterator.from_structure(val_data[t].data.output_types,
                                       val_data[t].data.output_shapes))
            
            next_batch_sr = iterator_source_train.get_next()
            next_batch_tr = iterator_target_train.get_next()
            
            next_batch_val = []
            for t in range(9):
                next_batch_val.append(iterator_val[t].get_next())
            
        train_batches_per_epoch = int(np.floor(train_source_data.data_size/batch_size))
        val_batches_per_epoch = int(np.ceil(val_data[0].data_size / batch_size))
        
        training_source_init_op = iterator_source_train.make_initializer(train_source_data.data)
        training_target_init_op = iterator_target_train.make_initializer(train_target_data.data)
        
        validation_init_op = []
        for t in range(9):
            validation_init_op.append(iterator_val[t].make_initializer(val_data[t].data))
        
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
                
        self.load_initial_weight(sess)
        
        print(config)
        print(init_lr)
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
                b = 1.0 * epochs / training_epochs
                #print(learning_rate)
                alpha = 2.0 / (1.0 + np.exp(-10.0 * epochs / training_epochs)) - 1.0 
                #(1 - np.exp(-10.0 * epochs / training_epochs)) / (1 + np.exp(-10.0 * epochs / training_epochs)) 
                
                x1, y1 = sess.run(next_batch_sr)
                x2, y2 = sess.run(next_batch_tr)                
                
                #print(times, np.mean(x1), np.mean(x2))
                _, total_loss, pred_loss, M_loss = sess.run([optimize, loss, prediction_loss, adver_loss], feed_dict = {self.source: x1, self.target : x2,
                                            one_hot_labels : y1, lr : learning_rate, self.keep_prob : self.KEEP_PROB_TRAINING, self.lamda : alpha, self.beta : b})
                
                L_pred = L_pred + pred_loss
                L_total = L_total + total_loss
                L_MMD = L_MMD + M_loss
    
            if not (epochs % 10 == 9):
                continue
            
            #print(sess.run([v for v in tf.trainable_variables() if 'conv1/weights' in v.name]))
            print(L_pred * 1.0 / train_batches_per_epoch)
            print(L_MMD / train_batches_per_epoch)
            print("TIMES: ", epochs, " LOSS: ", L_total * 1.0 / train_batches_per_epoch)
            
            test_acc = 0.
            num = 0.
            for t in range(9):
                sess.run(validation_init_op[t])
            
            for _ in range(val_batches_per_epoch):
                val_input = 0
                for t in range(9):    
                    if target_file == './testtxt/amazon.txt':
                        if t != 4:
                            continue
                    img_batch, label_batch = sess.run(next_batch_val[t])
                #img_batch = np.reshape(img_batch, [-1 ,height, width, 3])
                #print(img_batch.size)
                    logits = sess.run([tf.nn.softmax(self.logits, axis = -1)], feed_dict={self.source: img_batch, self.target : np.zeros(shape = (0,height, width,3), dtype = np.float32), 
                                       self.keep_prob : self.KEEP_PROB_VALIDATION, self.beta : 1.0})
                                    
                    val_input += logits[0]
                
                acc, num_ = sess.run([accuracy, accunum], feed_dict={prob: val_input, one_hot_labels: label_batch})
                test_acc += acc
                num += num_
            
            print("TIMES: ", epochs, " VALIDATION: ", test_acc / num)
            best = max(best, test_acc / num)
            
        print (best)

if source_file == './testtxt/dslr.txt':
    training_epochs = 800
elif source_file == './testtxt/webcam.txt' :
    training_epochs = 600
else:
    training_epochs = 400 
    
net = AdversarialNet(tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]), tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3]))
net.training(source_file, target_file, training_epochs = training_epochs)