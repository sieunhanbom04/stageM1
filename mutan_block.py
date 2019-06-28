#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:42:30 2019

@author: quoctung
"""

import tensorflow as tf
import numpy as np

#The function that allows us to do block multiplication by index

def block_multiplication(d, a, index):
    """
    d is a 4 dimension tensor [r, l, m, n] or [r, m, l, n] or [r, m, n, l] (depending on index = 1, 2, 3)
    a is a 3 dimension tensor [r, i, l]
    result is a 4 dimension tensor [r, i, m, n]
    """
    
    a_shape = tf.shape(a)
    perm = np.arange(4)
    perm[1] = index
    perm[index] = 1
    
    if index != 1:
        d = tf.transpose(d, perm = perm)
    
    d_shape = tf.shape(d)
    d = tf.reshape(d, [d_shape[0], d_shape[1], d_shape[2] * d_shape[3]])
    result = tf.matmul(a, d)
    result = tf.reshape(result, [d_shape[0], a_shape[1], d_shape[2], d_shape[3]])
    
    if index != 1:
        result = tf.transpose(result, perm = perm)
    return result

def mutan_block(x, y, x_dim, y_dim, proj_dim, reg_dim, domain = 'adversarial'):
    r = reg_dim[0]
    with tf.variable_scope(domain) as scope:
        """
        Create tf variables for the weights of D, A, B, C. 
        block = \sum_{i = 1, .., r} D_r x A_r x B_r x C_r
        """      
        d = tf.get_variable('D', shape = reg_dim)
        a = tf.get_variable('A', shape = [r, x_dim, reg_dim[1]]) 
        b = tf.get_variable('B', shape = [r, y_dim, reg_dim[2]]) 
        c = tf.get_variable('C', shape = [r, proj_dim, reg_dim[3]]) 
        
        block = block_multiplication(d, a, 1)
        block = block_multiplication(block, b, 2)
        block = block_multiplication(block, c, 3)
        block = tf.reduce_sum(block, axis = 0)
        
        y = tf.expand_dims(y, 1)
        print(y)
        result = tf.squeeze(tf.matmul(y, tf.tensordot(x, block, [[1], [0]])), axis = 1)
    return result