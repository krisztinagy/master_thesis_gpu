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
flags.DEFINE_integer('num_epochs', cfg.hyperparameters['num_epochs'], 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', cfg.hyperparameters['batch_size'], 'Batch size.')
flags.DEFINE_integer('image_pixels', cfg.image['height'] * cfg.image['width'] * cfg.image['channels'], 'Number of pixels in image')
flags.DEFINE_integer('image_height', cfg.image['height'], 'Height of image')
flags.DEFINE_integer('image_width', cfg.image['width'], 'Width of image')

flags.DEFINE_integer('num_classes', cfg.dataset['num_categories'], 'Number of classes')
flags.DEFINE_integer('num_threads', cfg.hyperparameters['num_threads'], 'Number of threads')

flags.DEFINE_string('results_dir', cfg.directory['results'] + '/', 'Directory with the training data.')
flags.DEFINE_string('model_dir', cfg.model['model_import'] + '/', 'Directory for storing model and results')
flags.DEFINE_string('tensorboard_dir', cfg.directory['tensorboard'], 'Data directory for storing tensorboard logs')

flags.DEFINE_string('tfrecords_train_dir', cfg.directory['tfrecords_train'], 'Data directory for storing tfRecords')
flags.DEFINE_string('tfrecords_test_dir', cfg.directory['tfrecords_test'], 'Data directory for storing tfRecords')
    
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
  #image_resized.set_shape([FLAGS.image_pixels])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image_flatten = tf.cast(image_flatten, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar and handle offset
  label = tf.cast(features['image/class/label'], tf.int32) - 1

  return image_flatten, label
  
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
  
  files = []
  
  for file in os.listdir(record_dir):
    file_path = os.path.join(record_dir, file)
    files.append(file_path)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into FLAGS.batch_size batches. Internally uses a RandomShuffleQueue.)
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=FLAGS.num_threads,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels 
    
# def export_graph(weights_path, checkpoint_folder, output_path):
    # network = model.Vgg19(weights_path)
    # graph_to_export = tf.Graph()
    # with tf.Session(graph=graph_to_export) as session:
        # graphdef = network.build_graph_for_export(session, checkpoint_folder)
        # with open(output_path, "wb") as file:
            # file.write(graphdef.SerializeToString())

###INFERENCE GRAPH - implemented in separate python module in folder 'models'

###TRAINING GRAPH - implemented in separate python module in folder 'loss_functions'

######################################################
######################################################
################       TRAINING        ###############
######################################################
######################################################    
def run_training(log_dir, last_checkpoint):

    # Build checkpoint data
    checkpoint = FLAGS.results_dir + FLAGS.model_dir + 'check'
    checkpoint_meta = FLAGS.results_dir + FLAGS.model_dir + 'check.meta'
    
    print('Checkpoint: %s' % checkpoint)
    print('Checkpoint meta: %s' % checkpoint_meta)
    
    # Tell TensorFlow that the model will be built into the default Graph.
    treedom_graph=tf.Graph()
    with treedom_graph.as_default():
    
        #Generate placeholders for images and labels
        images_placeholder=tf.placeholder(tf.float32)
        labels_placeholder=tf.placeholder(tf.int32)
        #Remember these operands
        tf.add_to_collection("images", images_placeholder)
        tf.add_to_collection("labels", labels_placeholder)
        
        # Build a Graph that computes predictions from the inference model.
        network = model.Vgg19(cfg.model['vgg19_pretrained'], trainable=True)
        logits = network.inference_graph(images_placeholder, True, FLAGS.image_pixels, FLAGS.num_classes)
        tf.add_to_collection("logits", logits)
        tf.summary.histogram("output", logits)
        print('LOGITS SHAPE:')
        print(logits)
        # tf.add_to_collection("logits0", logits[0])
        # tf.add_to_collection("logits1", logits[1])
        # tf.add_to_collection("logits2", logits[2])
        # tf.add_to_collection("logits3", logits[3])
        # tf.add_to_collection("logits4", logits[4])
        # tf.add_to_collection("logits5", logits[5])
        # tf.add_to_collection("logits6", logits[6])
        # tf.add_to_collection("logits7", logits[7])
        tf.summary.scalar("output00", logits[0][0])
        tf.summary.scalar("output01", logits[0][1])
        # tf.summary.scalar("output1", logits[1])
        # tf.summary.scalar("output2", logits[2])
        # tf.summary.scalar("output3", logits[3])
        # tf.summary.scalar("output4", logits[4])
        # tf.summary.scalar("output5", logits[5])
        # tf.summary.scalar("output6", logits[6])
        # tf.summary.scalar("output7", logits[7])
        
        #create images and labels
        images, labels = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        
        #create train and loss op
        train_op, loss = loss_function.training_graph(logits, labels_placeholder, FLAGS.learning_rate)
        tf.summary.scalar("Loss", loss)
        
        folder = FLAGS.results_dir + FLAGS.model_dir + FLAGS.tensorboard_dir
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        print('PATH OF ALL TENSORBOARD FILES')
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            print(file_path)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
        
        merged_summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(folder, treedom_graph)
        
        #ititalize global and local variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        #Create a saver for writing training checkpoints
        saver=tf.train.Saver(max_to_keep=1)
        
    with tf.Session(graph=treedom_graph) as sess:
    
        #initialize all the variables by running the op
        sess.run(init)
        
        # Restore checkpoint if we trained this model already
        if not last_checkpoint == 0:
            path = tf.train.latest_checkpoint(log_dir)
            saver.restore(sess, path)
            print('loaded from checkpoint YAAAAAAY')
            #saver.restore(sess, checkpoint)
        
        #variable for tracking losses - to be displayed in losses.png
        losses = []
        smoothed_losses = [None] * 100
        
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print("Starting threading")
        
        iteration = 0
        last_loss = 1
        try:
            while not coord.should_stop():
                start_time = datetime.datetime.now()
                
                image, label = sess.run([images, labels])
                    
                # Calculate loss
                _, loss_value, summary = sess.run([train_op, loss, merged_summaries], feed_dict={images_placeholder: image, labels_placeholder:label})
                last_loss = loss_value
                
                # Logging information
                losses.append(loss_value)
                writer.add_summary(summary, iteration)
                
                # Stats
                smoothed_losses[iteration % 100] = loss_value
                duration = datetime.datetime.now() - start_time
                
                # Print overview
                print("Iteration: %d, Loss:%.6f" % (iteration, loss_value))
                
                if iteration == 100:
                    duration = datetime.datetime.now() - start_time
                    images_per_minute = 100 * FLAGS.batch_size / ( duration.total_seconds() / 60 )
                    mylog.logging_general( log_dir, 'log_train', '%.3f images processed per minute' % images_per_minute )
                
                if iteration % 100 == 0 and not iteration == 0:
                    
                    smoothed_loss = np.sum(smoothed_losses) / 100
                    mylog.loss_log(log_dir, iteration, loss_value, smoothed_loss ) 
                
                if iteration % 1000 == 0 and not iteration == 0:

                    print('Saving checkpoint')
                    saver.save(sess, checkpoint, global_step = last_checkpoint + iteration)
                    saver.export_meta_graph(checkpoint_meta)
                    print("Finished saving checkpoint")
                    
                    smoothed_loss = np.sum(smoothed_losses) / 100
                    mylog.logging_step(log_dir, iteration, smoothed_loss, last_checkpoint)

                iteration += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d iterations.' % (FLAGS.num_epochs, iteration))
            f.write('Done training for %d epochs, %d iterations.\n' % (FLAGS.num_epochs, iteration))
            f.write('Final loss value: %.3f\n' % (last_loss))
        finally:
            coord.request_stop()
        
        # Wait for threads to finish.
        coord.join(threads)

        # model.Vgg19.save_npy(network, sess, "vgg19_save.npy")
        # Save training graph
        # tf.train.write_graph(tf.get_default_graph().as_graph_def(), FLAGS.results_dir + FLAGS.model_dir, "exported.pbtxt", as_text=True)
        
        # Save image
        fig=plt.figure()
        a1=fig.add_subplot(111)
        a1.plot(losses, label="losses")
        fig.savefig(FLAGS.results_dir + FLAGS.model_dir + 'losses.png')
        
        sess.close


log_dir, start_time, last_checkpoint = mylog.training_header()

run_training(log_dir, last_checkpoint)
#export_graph("vgg19.npy", "/home/devop/small_dataset/output/vgg-e1/", "exported_graph.pb")

mylog.training_footer(log_dir, start_time)