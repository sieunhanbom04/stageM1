import tensorflow as tf


import caffe
import sys 
import numpy as np

f1=open('./testfile', 'w')
print >> f1,"";
f1.close();
net = caffe.Net('models/bvlc_alexnet/deploy.prototxt', 
            '/media/ubuntu/sdcard/bvlc_alexnet.caffemodel', 
            caffe.TEST)
f1=open('./testfile', 'a')
for num in range(0,95):
    print >> f1,net.params['conv1'][0].data[num]
f1.close()
"""
class SquareTest(tf.test.TestCase):

	def testSquare(self):
		with self.test_session():
			x = tf.square([2, 3])
			self.assertAllEqual(x.eval(), [4, 9])


if __name__ == '__main__':
	tf.test.main()

"""
