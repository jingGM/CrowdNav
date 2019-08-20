import argparse
import time
import csv
import numpy as np
import rospy
import tensorflow as tf
import tensorlayer as tl



obs_shape = [0,0,0,60,80,1,3]
image= tf.placeholder(tf.float32, [None, obs_shape[6], obs_shape[3], obs_shape[4], obs_shape[5]], 'image_ph')

imagenet = tl.layers.InputLayer(image, name='image_input')
print(imagenet.outputs.shape)

imagenet = tl.layers.Conv3dLayer(imagenet, shape=(1, 5, 5, 1, 32), 
	strides=(1, 1, 2, 2, 1), padding='SAME', act=None, name='cnn3d_layer1')
print(imagenet.outputs.shape)
imagenet = tl.layers.Conv3dLayer(imagenet, shape=(3, 3, 3, 32, 64), 
	strides=(1, 3, 1, 1, 1), padding='SAME', act=None, name='cnn3d_layer2')
print(imagenet.outputs.shape)
imagenet = tl.layers.Conv3dLayer(imagenet, shape=(3, 3, 3, 64, 64), 
	strides=(1, 1, 2, 2, 1), padding='SAME', act=None, name='cnn3d_layer3')
print(imagenet.outputs.shape)
imagenet = tl.layers.Conv3dLayer(imagenet, shape=(3, 3, 3, 64, 64), 
	strides=(1, 1, 1, 1, 1), padding='SAME', act=None, name='cnn3d_layer4')
imagenet = tl.layers.Conv3dLayer(imagenet, shape=(3, 3, 3, 64, 128), 
    strides=(1, 1, 2, 2, 1), padding='SAME', act=None, name='cnn3d_layer5')
print(imagenet.outputs.shape)
imagenet = tl.layers.FlattenLayer(imagenet, name='imagefl')
print(imagenet.outputs.shape)
imagenet = tl.layers.DenseLayer(imagenet, n_units=960, act=tf.nn.relu, name='image_output')
image_output = imagenet.outputs
print(image_output.shape)
#print("=====")






'''
prev_layer : :class:`Layer`
    Previous layer.
shape : tuple of int
    Shape of the filters: (filter_depth, filter_height, filter_width, in_channels, out_channels).
strides : tuple of int
    The sliding window strides for corresponding input dimensions.
    Must be in the same order as the shape dimension.
padding : str
    The padding algorithm type: "SAME" or "VALID".
act : activation function
    The activation function of this layer.
W_init : initializer
    The initializer for the weight matrix.
b_init : initializer or None
    The initializer for the bias vector. If None, skip biases.
W_init_args : dictionary
    The arguments for the weight matrix initializer.
b_init_args : dictionary
    The arguments for the bias vector initializer.
name : str
    A unique layer name.
'''