#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:13:13 2019

@author: quoctung
"""
import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datagenerator import ImageDataGenerator

Iterator = tf.data.Iterator
valmode = True # Use full taget set for evaluation
display_step = 15

source_file = './testtxt3/wd_train.txt'
target_file = './testtxt/dslr_validate.txt'
size = len(open(source_file).readlines())

class MMD(AlexNet):
    def __init__(self, x, keep_prob, dimension, lamda, decay_l2 = 1e-4, model = './checkpoint/model', skip_layer = ['fc8'], train_layer = ['fc8_new', 'fc_adapt'], n_class = 1000, weight_path = "bvlc_alexnet.npy"):
        AlexNet.__init__(self, x, keep_prob, skip_layer, n_class, weight_path = "bvlc_alexnet.npy")
        self.rep_dim = dimension
        self.model = model
        self.lamda = lamda
        self.train_layers = train_layer
        self.decay_l2 = decay_l2
        self.KEEP_PROB_TRAINING = 0.5
        self.KEEP_PROB_VALIDATION = 1.0
        
    def create(self):
        super().create()
        
        #adaptation layer
        self.fc_adapt = self.fc(self.dropout7, 4096, self.rep_dim, name = 'fc_adapt')
        
        #final fc layer
        self.fc8_new = self.fc(self.fc_adapt, self.rep_dim, self.n_class, relu = False, name = 'fc8_new')
    
    def modify_accuracy(self, source, target, sess):
        result = sess.run(self.classifier, feed_dict = {self.X : source})
        prediction = np.argmax(result, axis = 1)
        #print (prediction)
        return accuracy_score(target, prediction)    
    
    def modify_data(self, data_source):
        index = np.arange(len(data_source[0])) 
        np.random.shuffle(index)
        data = data_source[0][index]
        training_label = data_source[1][index]
        training_data = []
        for i in range(len(data)):
            training_data.append(self.randomCrop(data[i]))
         
        return np.array(training_data), training_label
        
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
        
        XX = weight[:n_source, :n_source]
        YY = weight[n_source:, n_source:]
        XY = weight[:n_source, n_source:]
        YX = weight[n_source:, :n_source]
        
        result = tf.reduce_mean(XX + YY - XY - YX)
        #bandwidth_list = 
        return result
    
    def modify_training(self, source_file, target_file, loss_type = 'MMD', learning_rate = 1e-2, training_epochs = 400, batch_size = 300, flag = 0):
        super().create(flag)
        #self.classifier = self.fc(self.dropout7, 4096, self.n_class, relu = False, name = 'classifier')
        sess = tf.Session()
        
        var_list1 = [v for v in tf.trainable_variables() if not ("fc8" in v.name or "fc_adapt" in v.name) and "bias" not in v.name]
        var_list1b = [v for v in tf.trainable_variables() if not ("fc8" in v.name or "fc_adapt" in v.name) and "bias" in v.name]
        var_list2 = [v for v in tf.trainable_variables() if ("fc8" in v.name or "fc_adapt" in v.name) and "bias" not in v.name]
        var_list2b = [v for v in tf.trainable_variables() if ("fc8" in v.name or "fc_adapt" in v.name) and "bias" in v.name]
        
        print(var_list1)
        print(var_list1b)
        print(var_list2)
        print(var_list2b)
        
        var_list = var_list1+var_list1b+var_list2+var_list2b
        
        #labels = tf.placeholder(dtype = tf.int32, shape = batch_size, name = 'label')
        one_hot_labels = tf.placeholder(tf.float32, [None, self.n_class])
        lr = tf.placeholder(np.float32)

        classifier, _ = tf.split(self.fc8, [int(batch_size / 2), int(batch_size / 2)]) 
        if flag == 0:
            
            print ("RIGHT", source_file, target_file)
            labelfc6, unlabelfc6 = tf.split(self.fc6, [int(batch_size / 2), int(batch_size / 2)])
            labelfc7, unlabelfc7 = tf.split(self.fc7, [int(batch_size / 2), int(batch_size / 2)])
            labelfc8, unlabelfc8 = tf.split(self.fc8, [int(batch_size / 2), int(batch_size / 2)])
            
            if loss_type == 'MMD':
                MMD_loss = tf.reduce_mean(tf.square(tf.reduce_mean(labelfc6, 0) - tf.reduce_mean(unlabelfc6, 0)))
                MMD_loss += tf.reduce_mean(tf.square(tf.reduce_mean(labelfc7, 0) - tf.reduce_mean(unlabelfc7, 0)))
                MMD_loss += tf.reduce_mean(tf.square(tf.reduce_mean(labelfc8, 0) - tf.reduce_mean(unlabelfc8, 0)))
            else:
                print ("MK MMD")
                MMD_loss = self.gaussian_kernel(labelfc6, unlabelfc6) + self.gaussian_kernel(labelfc7, unlabelfc7) + self.gaussian_kernel(labelfc8, unlabelfc8)
            
        elif flag == 1:
            print ("WRONG", source_file, target_file)
            labeled, unlabeled = tf.split(self.fc_adapt, [int(batch_size / 2), int(batch_size / 2)])
            if loss_type == 'MMD':
                MMD_loss = tf.reduce_mean(tf.square(tf.reduce_mean(labeled, 0) - tf.reduce_mean(unlabeled, 0)))
            else:
                MMD_loss = self.gaussian_kernel(labeled, unlabeled)
            
        prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = classifier, labels = one_hot_labels))
        loss = prediction_loss + 0.0005 / 2 * tf.add_n([tf.nn.l2_loss(v) for v in var_list if 'bias' not in v.name]) + self.lamda * MMD_loss
        
        
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
        # Create optimizer and apply gradient descent to the trainable variables
        
        """
        optimizer1 = tf.train.AdamOptimizer(lr)
        optimizer1b = tf.train.AdamOptimizer(lr)
        optimizer2 = tf.train.AdamOptimizer(lr*10)
        optimizer2b = tf.train.AdamOptimizer(lr*10)
        """
        
        optimizer1 = tf.train.MomentumOptimizer(lr, momentum = 0.9)
        optimizer1b = tf.train.MomentumOptimizer(lr, momentum = 0.9)
        optimizer2 = tf.train.MomentumOptimizer(lr*10, momentum = 0.9)
        optimizer2b = tf.train.MomentumOptimizer(lr*10, momentum = 0.9)
        
        """
        optimizer1 = tf.train.GradientDescentOptimizer(lr)
        optimizer1b = tf.train.GradientDescentOptimizer(lr)
        optimizer2 = tf.train.GradientDescentOptimizer(lr*10)
        optimizer2b = tf.train.GradientDescentOptimizer(lr*10)
        
        """
        #train_op = optimizer1.apply_gradients(grads_and_vars=gradients)
        train_op1 = optimizer1.apply_gradients(grads_and_vars=gradients[:len(var_list1)])
        train_op1b = optimizer1b.apply_gradients(grads_and_vars=gradients[len(var_list1):len(var_list1+var_list1b)])
        train_op2 = optimizer2.apply_gradients(grads_and_vars=gradients[len(var_list1+var_list1b):len(var_list1+var_list1b+var_list2)])
        train_op2b = optimizer2b.apply_gradients(grads_and_vars=gradients[len(var_list1+var_list1b+var_list2):])
        optimize = tf.group(train_op1, train_op1b, train_op2, train_op2b)
        """
        opt1 = tf.train.GradientDescentOptimizer(learning_rate = lr * 10)
        opt2 = tf.train.GradientDescentOptimizer(learning_rate = lr)
    
        gradients = tf.gradients(loss, var_list)
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        grads1 = gradients[:len(var_list1)]
        grads2 = gradients[len(var_list1):]
        optimize1 = opt1.apply_gradients(zip(grads1, var_list1))
        optimize2 = opt2.apply_gradients(zip(grads2, var_list2))
        optimize = tf.group(optimize1, optimize2)
        """
        
        saver = tf.train.Saver()
        
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(one_hot_labels, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
            accunum = tf.shape(one_hot_labels)[0]
        
        with tf.device('/cpu:0'):
            tr_data = ImageDataGenerator(source_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=self.n_class,
                                     shuffle=True, size=size)
            val_data = ImageDataGenerator(target_file,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=self.n_class,
                                      shuffle=False, fulval=valmode)
            
    # create an reinitializable iterator given the dataset structure
            iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)

            next_batch = iterator.get_next()

# Ops for initializing the two different iterators
        train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
        val_batches_per_epoch = int(np.ceil(val_data.data_size / batch_size))
        
        training_init_op = iterator.make_initializer(tr_data.data)
        validation_init_op = iterator.make_initializer(val_data.data)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        self.load_initial_weight(sess)
        
        print(train_batches_per_epoch, self.lamda)
        
        best = 0
        
        for epochs in range(training_epochs):
            
            sess.run(training_init_op)
            
            #print(sess.run([tf.get_default_graph().get_tensor_by_name('conv1/weights:0')]))
            
            for times in range(train_batches_per_epoch):
                p = epochs / training_epochs + times/train_batches_per_epoch/training_epochs
                learning_rate = 0.001 / (1. + 0.001 * p)**0.75
                
                x1, y1 = sess.run(next_batch)
                
                _ , total_loss, pred_loss, mmd_loss = sess.run([optimize, loss, prediction_loss, MMD_loss], feed_dict = {self.X : x1, one_hot_labels : y1[:int(batch_size/2)], self.keep_prob : self.KEEP_PROB_TRAINING, 
                                          lr : learning_rate})
            
            if not (epochs % 5 == 4):
                continue
            
            print(pred_loss, mmd_loss)
            print("TIMES: ", epochs, " LOSS: ", total_loss)
            
            sess.run(validation_init_op)
            
            test_acc = 0.
            test_count = 0
            num = 0.
            for _ in range(val_batches_per_epoch):
                
                img_batch, label_batch = sess.run(next_batch)
                acc, num_ = sess.run([accuracy, accunum], feed_dict={self.X: img_batch,
                                                one_hot_labels: label_batch,
                                                self.keep_prob: 1.})
                test_acc += acc
                num += num_
            
            print("TIMES: ", epochs, " VALIDATION: ", test_acc / num)
            best = max(best, test_acc / num)
            
            """
            if epochs % 100 == 99:
                saver.save(sess, self.model)
            """
        print (best)
        
    
    """        
    def training(self, data_source, data_target, learning_rate = 1e-2, training_step = 3000, batch_source_size = 236, batch_target_size = 100):
        
        self.create()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.load_initial_weight(sess)
        
        var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.train_layers]
        var_list2 = [v for v in tf.trainable_variables() if not v.name.split('/')[0] in (self.train_layers + self.skip_layer)]
        
        var_list = var_list1 + var_list2
        
        labels = tf.placeholder(dtype = tf.int32, shape = batch_source_size, name = 'label')
        one_hot_labels = tf.one_hot(labels, depth = self.n_class)
        
        #prediction_loss = tf.reduce_mean(tf.nn.log_softmax(logits = self.fc8_new) * one_hot_labels)
        source_rep, target_rep =  tf.split(self.fc_adapt, [batch_source_size, batch_target_size])
        classifier, _ = tf.split(self.fc8_new, [batch_source_size, batch_target_size])
        
        prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = classifier, labels = one_hot_labels))
        MMD_loss = tf.reduce_mean(tf.square(tf.reduce_mean(source_rep, 0) - tf.reduce_mean(target_rep, 0)))
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list if 'biases' not in v.name])
        loss = prediction_loss + self.lamda * MMD_loss # + lossL2 * self.decay_l2
        
        opt1 = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        opt2 = tf.train.GradientDescentOptimizer(learning_rate = learning_rate * 0.1)
    
        gradients = tf.gradients(loss, var_list)
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        grads1 = gradients[:len(var_list1)]
        grads2 = gradients[len(var_list1):]
        optimize1 = opt1.apply_gradients(zip(grads1, var_list1))
        optimize2 = opt2.apply_gradients(zip(grads2, var_list2))
        optimize = tf.group(optimize1, optimize2)
        
        saver = tf.train.Saver()
        #saver.restore(sess, self.model)
        
        best = 0
        print (self.lamda)
        #saver.restore(sess, model)
        for times in range(training_step):
            training_data, training_label = self.data_preparation(data_source, data_target, batch_source_size, batch_target_size) 
            #np.concatenate([data_source[1][index_source], data_target[1][index_target]])
            
            #y = training_data[0]#.astype(np.uint8)
            #print (training_label[0])
            #plt.imshow(y)
            #plt.show()
            
            sess.run(optimize, feed_dict = {self.X : training_data, labels : training_label})
            
            if times % 10 == 9:
                pre_loss, M_loss, total_loss = sess.run([prediction_loss, MMD_loss, loss], feed_dict = {self.X : training_data, labels : training_label})    
                print("TIMES:", times, "     RESULT: ", total_loss , "    ACCURACY:", self.accuracy(training_data[:batch_source_size], training_label, sess))
                
                
            
            if times % 100 == 99:
                saver.save(sess, self.model)
            
            
        
    
    def evaluation(self, data_source, data_target, flag = False):
        if not flag:
            self.create()
        
        print(self.rep_dim)
        
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, self.model)
        
        print("ACCURACY:", self.accuracy(data_source, data_target, sess))
   
    def accuracy(self, source, target, sess):
        result = sess.run(self.fc8_new, feed_dict = {self.X : source})
        prediction = np.argmax(result, axis = 1)
        #print (prediction)
        return accuracy_score(target, prediction)
    
    def randomCrop(self, image, width = 227, height = 227):
        x = np.random.randint(0, image.shape[1] - width)
        y = np.random.randint(0, image.shape[0] - height)
        image = image[y:y+height, x:x+width]
        return image
    
    def data_preparation(self, data_source, data_target, batch_source_size, batch_target_size):
        index_source = np.random.choice(range(len(data_source[0])), batch_source_size)
        index_target = np.random.choice(range(len(data_source[0])), batch_target_size)
        
        #unsupervised learning
        data = np.concatenate([data_source[0][index_source], data_source[0][index_target]])
        training_label = data_source[1][index_source]
        
        training_data = []
        for i in range(len(data)):
            training_data.append(self.randomCrop(data[i]))
            
        return np.array(training_data), training_label

    """
     
#data_source = [np.load('amazon_feature.npy'), np.load('amazon_label.npy')]
#data_target = np.load('webcam_feature.npy')


mmd = MMD(tf.placeholder(dtype = tf.float32, shape = [None, 227, 227, 3]), tf.placeholder(dtype = tf.float32), 256, 1.0, n_class = 31, model = './checkpoint256_2/model')
mmd.modify_training(source_file, target_file, flag = 0, loss_type = 'MK')


#mmd = MMD(tf.placeholder(dtype = tf.float32, shape = [None, 227, 227, 3]), tf.placeholder(dtype = tf.float32), 256, 0.25 / 3.0, n_class = 31, model = './checkpoint256_2/model')
#mmd.modify_training(source_file, target_file, flag = 0)


"""
mmd = MMD(tf.placeholder(dtype = tf.float32, shape = [None, 227, 227, 3]), 1.0, 256, 0.25, n_class = 31, model = './checkpoint256_2/model')
validate_source = np.load('validate_webcam_feature.npy')
validate_target = np.load('validate_webcam_label.npy')

mmd.evaluation(validate_source, validate_target)
"""