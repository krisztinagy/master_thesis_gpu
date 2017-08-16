from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'loss_functions')
import time
import datetime
import math
import shutil

import numpy as np
import tensorflow as tf
import config as cfg
import my_log as mylog
exec "import %s as model" % ( cfg.model['model_import'] )
exec "import %s as loss_function" % ( cfg.model['loss_function_import'] )

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', cfg.hyperparameters['learning_rate'], 'Initial learning rate.')
flags.DEFINE_integer('num_epochs_eval', cfg.hyperparameters['num_epochs_eval'], 'Number of epochs to run tester.')
flags.DEFINE_integer('batch_size', cfg.hyperparameters['batch_size'], 'Batch size.')
flags.DEFINE_integer('image_pixels', cfg.image['height'] * cfg.image['width'] * cfg.image['channels'], 'Number of pixels in image')
flags.DEFINE_integer('image_height', cfg.image['height'], 'Height of image')
flags.DEFINE_integer('image_width', cfg.image['width'], 'Width of image')

flags.DEFINE_integer('num_classes', cfg.dataset['num_categories'], 'Number of classes')
flags.DEFINE_integer('num_threads', cfg.hyperparameters['num_threads'], 'Number of threads')

flags.DEFINE_string('results_dir', cfg.directory['results'] + '/', 'Directory with the training data.')
flags.DEFINE_string('model_dir', cfg.model['model_import'] + '_' + cfg.model['dataset'] + cfg.model['model_dir_special_note'] + '/', 'Directory for storing model and results')
flags.DEFINE_string('tensorboard_dir', cfg.directory['tensorboard'], 'Data directory for storing tensorboard logs')

flags.DEFINE_string('tfrecords_train_dir', '/pqry/pqry/data/' + 'tfrecords_' + cfg.model['dataset'] + '/train', 'Data directory for storing tfRecords')
flags.DEFINE_string('tfrecords_test_dir', '/pqry/pqry/data/' + 'tfrecords_' + cfg.model['dataset'] + '/test', 'Data directory for storing tfRecords')
    
######################################################
# Reads tfRecords files
#  Args:
#    filename_queue: the queue containing the files to be processed
#  Returns:
#    image: flattened image data tf.float32 [-0.5 0.5]
#    label: the label of the processed image, tf.int32 (0, 1, etc.)
######################################################
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image/height':tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      'image/class/label': tf.FixedLenFeature([], tf.int64),
      'image/filename': tf.FixedLenFeature([],tf.string),
      'image/encoded': tf.FixedLenFeature([], tf.string),
  })

  # Convert from a scalar string tensor to a uint8 tensor with shape
  # [FLAGS.image_pixels].
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  
  height = tf.cast(features['image/height'], tf.int32)
  width = tf.cast(features['image/width'], tf.int32)
  print('height shape:')
  print(tf.shape(height))
  #image.set_shape(height*width)
  
  image_reshaped = tf.reshape(image, [height, width, 1])
  image_resized = tf.image.resize_images(image_reshaped, [FLAGS.image_height, FLAGS.image_width])
  image_flatten = tf.reshape(image_resized, [FLAGS.image_height * FLAGS.image_width])
  image_flatten.set_shape([FLAGS.image_pixels])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image_flatten = tf.cast(image_flatten, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar and handle offset
  label = tf.cast(features['image/class/label'], tf.int32) - 1
  
  filename = tf.cast(features['image/filename'], tf.string)

  return image_flatten, label, filename
  
######################################################
# Reads input data num_epochs times.
#  Args:
#    train: Selects between the training (True) and validation (False) data.
#    batch_size: Number of examples per returned batch.
#    num_epochs: Number of times to read the input data, or 0/None to train forever.
#  Returns:
#    A tuple (images, labels), where:
#    * images is a float tensor with shape [batch_size, FLAGS.image_pixels] in [-0.5, 0.5].
#    * labels is an int32 tensor with shape [batch_size] with the true label, in [0, FLAGS.num_classes).
#    Note that an tf.train.QueueRunner is added to the graph, which must be run using e.g. tf.train.start_queue_runners().
######################################################
def inputs(train, batch_size, num_epochs):

  if not num_epochs: num_epochs = None
  
  record_dir = FLAGS.tfrecords_train_dir if train else FLAGS.tfrecords_test_dir
  
  all_files = []
  files = []
  
  for file in os.listdir(record_dir):
    file_path = os.path.join(record_dir, file)
    all_files.append(file_path)

  for i in range( int( len(all_files) * cfg.dataset['percentage_to_use'])):
    files.append(all_files[i])
    
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue
    image, label, filename = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into FLAGS.batch_size batches. Internally uses a RandomShuffleQueue.)
    images, sparse_labels, filenames = tf.train.shuffle_batch(
        [image, label, filename], batch_size=batch_size, num_threads=FLAGS.num_threads,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels , filenames

###INFERENCE GRAPH - implemented in separate python module in folder 'models'

###TRAINING GRAPH - implemented in separate python module in folder 'loss_functions'

######################################################
######################################################
################       TRAINING        ###############
######################################################
######################################################    
def run_evaluation(log_dir):
    
    # Tell TensorFlow that the model will be built into the default Graph.
    mygraph=tf.Graph()
    with mygraph.as_default():
    
        #Generate placeholders for images and labels
        images_placeholder=tf.placeholder(tf.float32)
        labels_placeholder=tf.placeholder(tf.int32)
        filenames_placeholder=tf.placeholder(tf.string)
        #Remember these operands
        tf.add_to_collection("images", images_placeholder)
        tf.add_to_collection("labels", labels_placeholder)
        tf.add_to_collection("filenames", labels_placeholder)
        
        # Build a Graph that computes predictions from the inference model.
        network = model.Vgg19(cfg.model['vgg19_pretrained'], trainable=True)
        logits = network.inference_graph(images_placeholder, True, FLAGS.image_pixels, FLAGS.num_classes)
        tf.add_to_collection("logits", logits)
        #tf.summary.histogram("output", logits)
        
        print("Graph built")
        
        #create images and labels
        images_eval, labels_eval, filenames_eval = inputs(train=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs_eval)
	    
	    #create eval op
        all = tf.nn.top_k(logits)
        eval_op_values, eval_op_indices = tf.nn.top_k(logits)
        
        #ititalize global and local variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())        
        sess = tf.Session()
        
        sess.run(init)
        print("initialized")
        
        print('logdir path: %s' % log_dir)
        path = tf.train.latest_checkpoint(log_dir)
        saver=tf.train.Saver()
        saver.restore(sess, path)
        
        print("Parameters loaded from checkpoint")
        
        #variable for tracking losses - to be displayed in losses.png
        predictions = []
        labels = []
        filenames = []
        
        step = 0
        total_error = 0
        total_very_bad = 0
        
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            while not coord.should_stop():
                start_time = time.time()
                image, label, filename = sess.run([images_eval, labels_eval, filenames_eval])
                all_data, logit, pred_values, prediction = sess.run([all, logits, eval_op_values, eval_op_indices], feed_dict={images_placeholder: image, labels_placeholder: label})
                prediction = prediction.flatten()
                
                labels = labels + label.tolist()
                predictions = predictions + prediction.tolist()
                filenames = filenames + filename.tolist()
                
                duration = time.time() - start_time
                abs_error = list(np.absolute(label - prediction))
                
                almost_correct = 0
                for i in range(len(prediction)):
                    diff = np.absolute( prediction[i] - label[i] )
                    if prediction[i] == 0 and ( label[i] == 1 or label[i] == 2 or label[i] == 4 ): almost_correct += 1
                    if prediction[i] == 1 and ( label[i] == 0 or label[i] == 5 or label[i] == 5 ): almost_correct += 1
                    if prediction[i] == 2 and ( label[i] == 3 or label[i] == 0 or label[i] == 6 ): almost_correct += 1
                    if prediction[i] == 3 and ( label[i] == 2 or label[i] == 1 or label[i] == 7 ): almost_correct += 1
                    if prediction[i] == 4 and ( label[i] == 5 or label[i] == 6 or label[i] == 0 ): almost_correct += 1
                    if prediction[i] == 5 and ( label[i] == 4 or label[i] == 7 or label[i] == 1 ): almost_correct += 1
                    if prediction[i] == 6 and ( label[i] == 7 or label[i] == 4 or label[i] == 2 ): almost_correct += 1
                    if prediction[i] == 7 and ( label[i] == 6 or label[i] == 5 or label[i] == 3 ): almost_correct += 1
                    
                
                correct = abs_error.count(0)
                error = ( FLAGS.batch_size - correct )
                very_bad = error - almost_correct
                print("Error: %.3f Step: %d" % ( error / FLAGS.batch_size, step))
                step += 1
                total_error += error
                total_very_bad += very_bad
                
        except tf.errors.OutOfRangeError:
            print('Done evaluation for %d epochs, %d steps.' % (FLAGS.num_epochs_eval, step))
            mylog.logging_general(log_dir, 'log_test', 'Done evaluation for %d epochs, %d steps.\n' % (FLAGS.num_epochs_eval, step))
            print("done")
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            
        total_error = total_error / ( FLAGS.batch_size * step)
        total_very_bad = total_very_bad / ( FLAGS.batch_size * step)
        
        print('Average error during evaluation of %d steps: %.3f percent. Very bad results: %3f percent.' % (step, total_error * 100, total_very_bad * 100))
        mylog.logging_general(log_dir, 'log_test', 'Average error during evaluation of %d steps: %.6f (%.3f percent). Very bad results: %3f percent.\n' % (step, total_error, total_error * 100, total_very_bad * 100))
        
        # Wait for threads to finish.
        coord.join(threads)
    
        f = open(log_dir + '/test_results', 'a+')
        
        for element in range( len(predictions) ):
            f.write( '%s, %d, %d,\n' % (filenames[element], labels[element], predictions[element]) )
            
        f.close()
        
        sess.close()
		
log_dir, start_time = mylog.testing_header()

run_evaluation(log_dir)

mylog.testing_footer(log_dir, start_time)