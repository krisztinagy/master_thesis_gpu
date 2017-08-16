import tensorflow as tf

import numpy as np
from functools import reduce
import math

import sys
#sys.path.insert(0, '/home/nagy729krisztina/M4_treedom')
import config as cfg

class Vgg19:

	######################################################
	# Load variables form npy to build the VGG
	#  Args:
	#    vgg19_npy_path: path to the npy file storing the trained parameters
	#    trainable: boolean, if True, the paramaters can be trained and dropout will be turned on
	#	 dropout: chance of dropout for fully connected layers
	#  Returns:
	#    None
	######################################################
    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

	######################################################
	# Load variables form npy to build the VGG
	#  Args:
	#    rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
	#    train_mode: a boolean, if True, dropout will be turned on
	#  Returns:
	#    self.output: the output of the network, the result of classification
	######################################################
    def inference_graph(self, grayscale, train_mode=None, image_pixels = cfg.image['height']*cfg.image['width'], num_classes = cfg.dataset['num_categories']):
	
        grayscale_scaled = grayscale * 255.0
        
        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal([image_pixels, cfg.fully_connected['hidden1']], 0.0, 0.001))
            biases = tf.Variable(tf.zeros([cfg.fully_connected['hidden1']]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(grayscale_scaled, weights) + biases)
            
        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([cfg.fully_connected['hidden1'], cfg.fully_connected['hidden2']],0.0, 0.001))
            biases = tf.Variable(tf.zeros([cfg.fully_connected['hidden2']]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
            
        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([cfg.fully_connected['hidden2'], cfg.dataset['num_categories']],0.0, 0.001))
            biases = tf.Variable(tf.zeros([cfg.dataset['num_categories']]), name='biases')
            logits = tf.matmul(hidden2, weights) + biases
            
        return logits
        
        