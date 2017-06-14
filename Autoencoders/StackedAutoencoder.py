'''
Example of a stacked autoencoder using MNIST dataset
'''
import tensorflow as tf
import numpy as np

################## Load Data ###############
# Data is already scaled in [0, 1] range
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

############ Graph Construction ############

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()


n_inputs = 28 * 28 # for MNIST
n_hidden1 = 2*300
n_hidden2 = 150 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001
l2_reg = 0.001

X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='Inputs')

from tensorflow.contrib.layers import fully_connected
with tf.contrib.framework.arg_scope([fully_connected],activation_fn=tf.nn.elu,
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
    hidden1 = fully_connected(inputs=X, num_outputs=n_hidden1)
    hidden2 = fully_connected(inputs=hidden1, num_outputs=n_hidden2)
    hidden3 = fully_connected(inputs=hidden2, num_outputs=n_hidden3)
    output = fully_connected(inputs=hidden3, num_outputs=n_outputs,
                             activation_fn=None)

# Get the losses
reconstruction_loss = tf.reduce_mean(tf.square(X - output), name='mse_loss')
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_loss = tf.norm(reg_losses)
loss = reg_losses + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

############### Execution Phase ##################
n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        loss_val = 0.
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: X_batch})

        print('===== Epoch: {0} ====='.format(epoch))
        [loss_train, reg_loss_val] = sess.run([reconstruction_loss, reg_loss], feed_dict={X: mnist.train.images})
        loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.validation.images})
        print('Train: Reconstruction MSE = {0:.4f}, Regularization: {1:.4f}'.format(loss_train, reg_loss_val))
        print('Validation MSE: {0:.4f}'.format(loss_val))

    # Print test loss
    loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.test.images})
    print('Test MSE: {0:.4f}'.format(loss_val))







