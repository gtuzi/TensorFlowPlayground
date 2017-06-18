'''
Construct a simple autoencoder where with sparsity cost
'''

import tensorflow as tf
import numpy as np



################## Load Data ###############
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
################### Parameters #############
learning_rate = 0.01
sparsity_target = 0.01
sparsity_weight = 0.2*3

n_inputs = 28 * 28 # for MNIST
n_hidden1 = 1000 # codings
n_output = n_inputs
learning_rate = 0.01
l2_reg = 0.0000000001

n_epochs = 100

# batch size must be very large for the sparsity measure to be meaningful
batch_size = 1000

######## Construction Phase ################
'''
Construct a simple autoencoder with 3 layers
'''
def kl_divergence(p, q):
    return p*tf.log(p/q)+(1-p)*tf.log((1-p)/(1-q))


def construct_layer(X, n_in, n_out,
                    initializer = tf.contrib.layers.variance_scaling_initializer(),
                    activation=tf.nn.sigmoid,
                    regularizer = None,
                    layer_name=''):
    with tf.variable_scope('layer' + layer_name):
        W = tf.get_variable(name='W' + layer_name,
                            initializer=initializer([n_in, n_out]), # also defines shape
                            dtype=tf.float32,
                            regularizer=regularizer)

        b = tf.get_variable(name='b' + layer_name,
                            dtype=tf.float32,
                            initializer=tf.zeros(shape=n_out))
        z = tf.matmul(X, W) + b

        if activation is not None:
            return activation(z)
        else:
            return z




X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs),name='X')
l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)

layer1 = construct_layer(X, n_inputs, n_hidden1, regularizer = l2_reg, layer_name='layer1')
output = construct_layer(layer1, n_hidden1, n_output, activation=None, regularizer = l2_reg, layer_name='output')

# The actual activation probability is the propability that the
# neuroan is active. Because the activation function is 0-1, this is
# the expected probability, over the batch, that the neuron will be active
sparsity = tf.reduce_mean(layer1, axis=0)
sparsity_loss = tf.reduce_sum(kl_divergence(p=tf.constant(sparsity_target, dtype=tf.float32), q=sparsity))
reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
reconstruction_loss = tf.reduce_mean(tf.square(X - output), name='recon_loss_mse')

loss = reconstruction_loss + reg_loss + sparsity_weight*sparsity_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)



############# Execute: Train ####################

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        loss_val = 0.
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})

        print('===== Epoch: {0} ====='.format(epoch))
        loss_train, reg_loss_train, sparsity_loss_train = sess.run([reconstruction_loss, reg_loss, sparsity_loss],
                                                                   feed_dict={X: mnist.train.images})

        print(str('Reconstruction MSE = {0:.4f}, ' +
              'Regularization: {1:.4f}, '+
              'Sparsity: {2:.4f}').format(loss_train,reg_loss_train,sparsity_loss_train))

        loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.validation.images})
        print('Validation MSE: {0:.4f}'.format(loss_val))