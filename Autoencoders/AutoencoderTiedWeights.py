'''
Example showing the implementation of a stacked autoencoder with
tied weights.

Observation: Using MNIST, performance of the tied autoencoder is superior
to the plain stacked autoencoder
'''

import tensorflow as tf
import numpy as np

################## Load Data ###############
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")


############ Plot ##########################
import matplotlib.pyplot as plt

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def show_reconstructed_digits(X, outputs, n_test_digits = 2):
    X_test = mnist.test.images[:n_test_digits]
    outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
    plt.waitforbuttonpress()


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
n_output = n_inputs
learning_rate = 0.01
l2_reg = 0.0001

activation = tf.nn.elu
l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='X')

W1_init = initializer([n_inputs, n_hidden1])
W2_init = initializer([n_hidden1, n_hidden2])


W1 = tf.Variable(initial_value=W1_init, dtype=tf.float32, name='weights1')
W2 = tf.Variable(initial_value=W2_init, dtype=tf.float32, name='weights2')

# Tied weights
W3 = tf.transpose(W2, name='weights4')
W4 = tf.transpose(W1, name='weights5')

b1 = tf.Variable(initial_value=tf.zeros(shape=n_hidden1), name='bias1')
b2 = tf.Variable(initial_value=tf.zeros(shape=n_hidden2), name='bias2')

# Biases are not tied
b3 = tf.Variable(initial_value=tf.zeros(shape=n_hidden3), name='bias3')
b4 = tf.Variable(initial_value=tf.zeros(shape=n_output), name='bias4')

# input --> hidden layer 1
layer1 = activation(tf.matmul(X, W1) + b1, name='layer1')
# hidden layer 1 --> codings
layer2 = activation(tf.matmul(layer1, W2) + b2, name='layer2_codings')
# codings --> hidden layer 3
layer3 = activation(tf.matmul(layer2, W3) + b3, name='layer3')
# hidden layer 3 --> output
output = tf.add(tf.matmul(layer3, W4), b4, name='output')


reconstruction_loss = tf.reduce_mean(tf.square(X - output), name='MSE')

# Note:
# ** Only weights are regularized
# ** Only the "variable" weights are regularized since
#    the other weights are tied to these ones
# ** Biases are not regularized
reg_loss = tf.norm([l2_reg(W1), l2_reg(W2)])
loss = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

################### Execution Phase ##########################

n_epochs = 10
batch_size = 150

sess = tf.InteractiveSession()
init.run()
for epoch in range(n_epochs):
    n_batches = mnist.train.num_examples // batch_size
    loss_val = 0.
    for iteration in range(n_batches):
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: X_batch})

    print('===== Epoch: {0} ====='.format(epoch))
    loss_train, reg_loss_train = sess.run([reconstruction_loss, reg_loss], feed_dict={X: mnist.train.images})
    loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.validation.images})
    print('Train: Reconstruction MSE = {0:.4f}, Regularization: {1:.4f}'.format(loss_train, reg_loss_train))
    print('Validation MSE: {0:.4f}'.format(loss_val))
    print('')

# Print final losses
test_loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.test.images})
valid_loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.validation.images})
train_loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.train.images})
print('Train: {0:.4f}, Valid: {1:.4f}, Test: {2:.4f} MSE'.format(train_loss_val, valid_loss_val, test_loss_val))

show_reconstructed_digits(X, output)

sess.close()



