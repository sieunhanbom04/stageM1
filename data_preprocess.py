#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:06:33 2019

@author: quoctung
"""

import numpy as np
import os
from scipy import ndimage, misc
import matplotlib.pyplot as plt

def load_category(filepath, h = 256, w = 256, channel = 3, num = 20):
    result = np.zeros((num, h, w, channel))
    folder = os.listdir(filepath) 
    index = np.random.choice(folder, size = num)
    
    for i in range(len(index)):
        file = index[int(i)]
        source = filepath + '/' + file
        img = plt.imread(source)
        img = misc.imresize(img, (w,h))
        result[i] = img
    
    return result

def load_category_txt(filepath, flag, num = 20):
    folder = os.listdir(filepath) 
    if flag:
        num = min(num, len(folder))
        index = np.random.choice(folder, size = num, replace = False)
    else:
        index = folder
        num = len(index)
        
    result = []
    
    for i in range(num):
        file = index[int(i)]
        result.append(filepath + '/' + file)
    
    return result   
        
def load_training_data(filepath, written_file,num = 20, flag = True, h = 256, w = 256):
    folder = os.listdir(filepath)
    result = []
    label = []
    print (folder)
    
    for i in range(len(folder)):
        source = filepath + '/' + folder[int(i)]
        x = load_category_txt(source, num = num, flag = flag)
        result = result + x
        label.append(np.full(len(x), i))
    
    label = np.concatenate(label)
    with open(written_file, "w") as text_file:
        for i in range(len(result)):
            text_file.write(result[i] + " " + "%d" % (label[i]) + "\n")
        text_file.close()

def load_data(filepath, num = 20):
    folder = os.listdir(filepath)
    result = []
    label = []
    print (folder)
    
    for i in range(len(folder)):
        source = filepath + '/' + folder[int(i)]
        x = load_category_txt(source, flag = True, num = num)
        result = result + x
        label.append(np.full(len(x), i))
    
    label = np.concatenate(label)
    return result, label

def load_semi_training_data(filepath1, filepath2, written_file, num1 = 20, num2 = 3):
    result1, label1 = load_data(filepath1, num1)
    result2, label2 = load_data(filepath2, num2)
    
    result = result1 + result2
    label = np.concatenate([label1, label2])
    
    with open(written_file, "w") as text_file:
        for i in range(len(result)):
            print(result[i] + " " + "%d" % (label[i]) + "\n")
            text_file.write(result[i] + " " + "%d" % (label[i]) + "\n")
        text_file.close()
    

#load_training_data('./domain_adaptation_images/dslr/images', './testtxt3/dslr_train.txt', num = 8)
load_training_data('/local/quoctung/domain_adaptation_images/dslr/images', './testtxt/dslr.txt', flag = False)

"""            
x, y = load_training_data(filepath = './domain_adaptation_images/webcam/images', num = 8, flag = True, h = 227, w = 227)
#x, y = load_training_data(filepath = './domain_adaptation_images/amazon/images', num = 20, flag = False, h = 256, w = 256)
np.save('webcam_feature',x)
np.save('webcam_label',y)
"""