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
        
        # Hidden 1
        with tf.name_scope('hidden'):
            weights = tf.Variable(tf.truncated_normal([image_pixels, cfg.fully_connected['hidden']],stddev=1.0 / math.sqrt(float(image_pixels))))
            biases = tf.Variable(tf.zeros([cfg.fully_connected['hidden']]), name='biases')
            hidden = tf.nn.relu(tf.matmul(grayscale, weights) + biases)
            
        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([cfg.fully_connected['hidden'], cfg.dataset['num_categories']], stddev=1.0 / math.sqrt(float(image_pixels))))
            biases = tf.Variable(tf.zeros([cfg.dataset['num_categories']]), name='biases')
            logits = tf.matmul(hidden, weights) + biases
            
        return logits
        