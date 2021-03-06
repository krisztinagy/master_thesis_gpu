import tensorflow as tf

import numpy as np
from functools import reduce

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
    def inference_graph(self, grayscale, train_mode=None, image_pixels = cfg.image['height']*cfg.image['width'], num_classes = 2):
	
        grayscale_scaled = grayscale * 255.0
        
        ### Dimension info: images_reshaped shape: expected to be [batch_size, 64, 64, 1]
        grayscale_reshaped = tf.reshape(grayscale_scaled, [-1,cfg.image['height'],cfg.image['width'],1], name="grayscale_reshaped" )
        print("grayscale_reshaped size:")
        print(grayscale_reshaped.get_shape())
        
        
        ### VGG Net definition ###
        #if load is False, trainable has to be True
        self.conv1_1 = self.first_conv_layer(grayscale_tensor, 1, 64, "conv1_1", load=True, trainable=False)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", load=True, trainable=False)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", load=True, trainable=False)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", load=True, trainable=False)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", load=True, trainable=False)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", load=True, trainable=False)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", load=True, trainable=False)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", load=True, trainable=False)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", load=True, trainable=False)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", load=True, trainable=False)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", load=True, trainable=False)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", load=True, trainable=False)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", load=True, trainable=False)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", load=True, trainable=False)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", load=True, trainable=False)
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", load=True, trainable=False)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        
        # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        input_nodes = pow( (cfg.image['height'] / 32), 2) * 512 
        self.fc6 = self.fc_layer(self.pool5, input_nodes, 1024, "fc6_modified", load=False, trainable=True)
        self.relu6 = tf.nn.relu(self.fc6)
        
        if train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 1024, 1024, "fc7_modified", load=False, trainable=True)
        self.relu7 = tf.nn.relu(self.fc7)
            
        if train_mode:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 1024, num_classes, "fc8_modified", load=False, trainable=True)
        
        self.output = tf.nn.softmax(self.fc8, name="prob")
        
        return self.output
		
	######################################################
	# Helper functions for defining layers
	######################################################
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, load, trainable):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, load, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
            
    def first_conv_layer(self, bottom, in_channels, out_channels, name, load, trainable):
        with tf.variable_scope(name):
            filt_all, conv_biases_all = self.get_conv_var(3, 3, out_channels, name, load, trainable)
            
            filt_tmp = tf.reshape(filt_all, [-1])
            filt_length = tf.size(filt_tmp) / 3
            f = filt_tmp[0:filt_length]
            filt = tf.reshape(f, [3, 3, 1, 64])
            
            # conv_biases_tmp = tf.reshape(conv_biases_all, [-1])
            # conv_biases_length = tf.size(conv_biases_tmp) / 3
            # c = conv_biases_all[0:conv_biases_length]
            # conv_biases = tf.reshape(c, [64])
            
            conv_biases = conv_biases_all

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name, load, trainable):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, load, trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

	######################################################
	# Sets the weights and biases for a convolutional layer
	#  Args:
	#    filter_size: size of conv filter (int, actual filter size is the square of this)
	#    in_size: number of input channels
	#    out_size: number of output channels
	#    name: name of layer
    #    load: whether to load the pretrained weights from file
    #    trainable: whether the weights can be modified during training
	#  Returns:
	#    filters: filter values for this layer
	#    biases: bias values for this layer
	######################################################
    def get_conv_var(self, filter_size, in_channels, out_channels, name, load, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return filters, biases

	######################################################
	# Sets the weights and biases for a fully connected layer
	#  Args:
	#    in_size: number of input nodes
	#    out_size: number of output nodes
	#    name: name of layer
    #    load: whether to load the pretrained weights from file
    #    trainable: whether the weights can be modified during training
	#  Returns:
	#    weights: weight values for this layer
	#    biases: bias values for this layer
	######################################################
    def get_fc_var(self, in_size, out_size, name, load, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable)
        
        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return weights, biases

		
	######################################################
	# Sets the weights or biases for one layer
	#  Args:
	#    initial_value: truncated normals of filter(for weights) or output(for biases) size
	#    name: name of the layer
	#    idx: 0 for weights, 1 for biases
	#    var_name: name of the entity: 'name' + '_filters' or 'name' + '_biases'
	#  Returns:
	#    var: the variable values read from the data structure (either for the weights or for the biases)
	######################################################
    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
    
    ######################################################
	# Save the updated(?) parameters
	#  Args:
	#    sess: session that runs the functions
	#    npy_path: the file to save the parameters to	 
	#  Returns:
	#    npy_path: returns the path to the file the parameters have been saved to
	######################################################
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path
        

	######################################################
	# Counts the parameters of the model
	#  Args:
	#    None
	#  Returns:
	#    count: number of parameters
	######################################################
    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count