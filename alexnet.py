
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc

class AlexNet(object):
    def __init__(self, x, keep_prob, skip_layer, n_class = 1000, weight_path = "bvlc_alexnet.npy"):
        self.X = x
        self.keep_prob = keep_prob
        self.weight_path = weight_path
        self.skip_layer = skip_layer
        self.n_class = n_class
        
    def conv(self, x, h, w, num_filters, stride_x, stride_y, name, padding = 'SAME', groups = 1):
        input_size = (int)(x.get_shape()[-1])
        
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
        
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape = [h, w, input_size/groups, num_filters])
            biases = tf.get_variable('biases', shape = [num_filters]) 
            
            if groups == 1:
                conv = tf.nn.conv2d(x, weights, strides = [1, stride_x, stride_y, 1], padding = padding)
            else:
                input_groups = tf.split(axis = 3, num_or_size_splits= groups, value = x)
                weight_groups = tf.split(axis = 3, num_or_size_splits = groups, value = weights)
                output_groups = [convolve(i,k) for i,k in zip(input_groups, weight_groups)]
                
                conv = tf.concat(axis = 3, values = output_groups)
                
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name = scope.name)
        
        return relu

    def fc(self, x, num_in, num_out, name, relu = True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape = [num_in, num_out], trainable = True)
            biases = tf.get_variable('biases', shape = [num_out], trainable = True)
            
            result = tf.nn.xw_plus_b(x, weights, biases, name = scope.name)
            
            if relu:
                relu = tf.nn.relu(result)
                return relu
            
            return result
    
    def max_pool(self, x, h, w, stride_x, stride_y, name, padding = 'SAME'):
        return tf.nn.max_pool(x, ksize = [1, h, w, 1], strides = [1, stride_x, stride_y, 1], padding = padding, name = name) 
    
    def lrn(self, x, name, radius = 2, alpha = 1e-05, beta = 0.75, k = 1.0):
        return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha, beta = beta, bias = k, name = name)
    
    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
    
    def create(self, flag = -1):
        #1st layer: convolution kernel = [11, 11, 3, 96], strides = [4, 4]
        self.conv1 = self.conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
        self.norm1 = self.lrn(self.conv1, name = 'norm1')
        self.pool1 = self.max_pool(self.norm1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
        
        #2nd layer: convolution kernel = [5, 5, 96, 256], strides = [1, 1]
        self.conv2 = self.conv(self.pool1, 5, 5, 256, 1, 1, name = 'conv2', groups = 2)
        self.norm2 = self.lrn(self.conv2, name = 'norm2')
        self.pool2 = self.max_pool(self.norm2, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')
        
        #3rd layer: convolution kernel = [3, 3, 256, 384], strides = [1, 1]
        self.conv3 = self.conv(self.pool2, 3, 3, 384, 1, 1, name = 'conv3')
        
        #4th layer: convolution kernel = [3, 3, 384, 384]
        self.conv4 = self.conv(self.conv3, 3, 3, 384, 1, 1, name = 'conv4', groups = 2)
        
        #5th layer: convolution kernel = [3, 3, 384, 256]
        self.conv5 = self.conv(self.conv4, 3, 3, 256, 1, 1, name = 'conv5', groups = 2)
        self.pool5 = self.max_pool(self.conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
        
        #6th layer: fully-connected w = [9216, 4096]
        temp = tf.reshape(self.pool5, shape = [-1, 6 * 6 * 256])
        self.fc6 = self.fc(temp, 6 * 6 * 256, 4096, name = 'fc6')
        self.dropout6 = self.dropout(self.fc6, self.keep_prob)
                
        #7th layer: fully-connected
        self.fc7 = self.fc(self.dropout6, 4096, 4096, name = 'fc7')
        self.dropout7 = self.dropout(self.fc7, self.keep_prob)      
        
        #8th layer: fully-connected
        #self.fc8 = self.fc(self.dropout7, 4096, self.n_class, relu = False, name = 'fc8')
        if flag == 0:
            self.fc8 = tf.layers.dense(self.dropout7, units=self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="fc8")
        elif flag == 1:
            self.fc_adapt = self.fc(self.dropout7, 4096, 256, name = 'fc_adapt')
            self.dropout8 = self.dropout(self.fc_adapt, self.keep_prob)
            self.fc8 = tf.layers.dense(self.dropout8, units = self.n_class, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), use_bias=True, name="fc8") 
            
    def load_initial_weight(self, session):
        
        weight_dicts = np.load(self.weight_path, encoding = 'bytes').item()
        
        for op_name in weight_dicts:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse = True):
                    for data in weight_dicts[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable = True)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable = True)
                            session.run(var.assign(data))
    
    def testing(self, image):
        self.create()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.load_initial_weight(sess)
        result = sess.run([tf.nn.softmax(self.fc8)], feed_dict = {self.X : image})
        print (np.argmax(result), np.max(result))
        
def preprocess(image_path):
    img = plt.imread(image_path)
    img = misc.imresize(img, (227, 227))
    r = np.zeros(img.shape)
    r = img
    #plt.imshow(r)
    #plt.show()
    #r = img / np.max(img)
    return np.array([r], dtype = np.float32)

"""
alex = AlexNet(tf.placeholder(tf.float32, name = 'dkm', shape = [1, 227, 227, 3]), 0.5, [])
alex.testing(preprocess('terrier.jpg'))
"""     