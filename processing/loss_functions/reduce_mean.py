import tensorflow as tf
import config as cfg

######################################################
# Build the training graph.
#  Args:
#    logits: Logits tensor, float - [FLAGS.batch_size, FLAGS.num_classes].
#	 labels: Labels tensor, int32 - [BATCH_SIZE], with values in [0, FLAGS.num_classes).
#	 learning_rate: The learning rate to use for gradient descent.
#  Returns:
#    train_op: The Op for training.
#    loss: The Op for calculating loss.
######################################################
def training_graph(logits, labels, learning_rate):
    
    # Create an operation that calculates loss.
    # labels_flat = tf.reshape(tf.cast(labels, tf.float32, name="label"), [-1])
    # logits_flat = tf.reshape(tf.cast(logits, tf.float32, name="logits"), [-1])
    # loss = tf.reduce_mean(tf.abs(labels_flat - logits_flat, name="loss"), name='xentropy_mean')
    labels = tf.to_int32(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
    
    # Create a variable to track the global step (iteration).
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    return train_op, loss