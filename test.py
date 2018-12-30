# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 18:26:31 2018

@author: Administrator
"""

import tensorflow as tf

input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')