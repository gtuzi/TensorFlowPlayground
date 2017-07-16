'''
Train Autoencoder networks one at a time. Then combine the final
network into one. Each of these steps are phases.
To achieve this, use a different TensorFlow graph for each phase.
At each phase, you build the outer layers, as you progress inwards.
Training data for the next phase becomes the training output
from the previous phase.
'''

import tensorflow as tf
import  numpy as np
################## Load Data ###############
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")


##################  Parameters ##############
n_inputs = 28 * 28 # for MNIST
n_hidden1 = 300
n_hidden2 = 150 # codings
n_hidden3 = n_hidden1
n_output = n_inputs
learning_rate = 0.001
l2_reg = 0.0001

n_epochs = 4
batch_size = 150


def reset_graph():
    tf.reset_default_graph()
    np.random.seed()

activation = tf.nn.elu
l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()


################## Phase 1 #################

ph1Graph = tf.Graph()
# Training data for the next autoencoder layer
L1_TrainData = None
L1_ValidData = None


# Weights for reconstruction
L1_Wvals = None
L1_bvals = None
Lout_Wvals = None
Lout_bvals = None
with ph1Graph.as_default():

    # The following source ops and variables will be added to this graph
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='X')

    # --- Layer 1 (Hidden)
    W1 = tf.Variable(dtype=tf.float32, initial_value=initializer([n_inputs, n_hidden1]), name='W1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden1), name='b1')
    layer1 = activation(tf.add(tf.matmul(X, W1), b1), name='layer1')

    # --- Output layer
    Wout = tf.Variable(dtype=tf.float32, initial_value=initializer([n_hidden1, n_output]), name='Wout')
    bout = tf.Variable(tf.zeros(shape=n_output), name='bout')
    # No non-linearity here. We're comparing to a certain output
    output = tf.add(tf.matmul(layer1, Wout), bout, name='output')


    reconstruction_loss = tf.reduce_mean(tf.square(X - output), name='mse')
    reg_losses = [l2_reg(W1), l2_reg(Wout)]
    reg_loss = tf.add_n(reg_losses)
    loss = tf.add_n([reconstruction_loss] + reg_losses)
    # loss = reconstruction_loss + reg_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session(graph=ph1Graph) as sess:
        sess.run(init)

        # Phase 1 data, is the actual data
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            loss_val = 0.
            for iteration in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})

            print('===== Phase 1 - Epoch: {0} ====='.format(epoch))
            loss_train, reg_loss_train = sess.run([reconstruction_loss, reg_loss], feed_dict={X: mnist.train.images})
            loss_val = sess.run(reconstruction_loss, feed_dict={X: mnist.validation.images})
            print('Train: Reconstruction MSE = {0:.4f}, Regularization: {1:.4f}'.format(loss_train, reg_loss_train))
            print('Validation MSE: {0:.4f}'.format(loss_val))

        # The weights will be used for reconstructing the final autoencoder
        L1_Wvals, L1_bvals = W1.eval(), b1.eval()
        Lout_Wvals, Lout_bvals = Wout.eval(), bout.eval()

        # The output of the hidden layer of this phase,
        # becomes training for the next phase's hidden layer
        L1_TrainData = sess.run(layer1, feed_dict={X: mnist.train.images})
        L1_ValidData = sess.run(layer1, feed_dict={X: mnist.validation.images})



################## Phase 2 #####################

graph2 = tf.Graph()
L2_Wvals = None
L2_bvals = None
L3_Wvals = None
L3_bvals = None
with graph2.as_default():

    L1_Dat = tf.placeholder(dtype=tf.float32, shape=(None, n_hidden1), name='L1_Dat')
    # The following variables will be added to this graph

    # ---- Coding layer
    W2 = tf.Variable(dtype=tf.float32, initial_value=initializer([n_hidden1, n_hidden2]), name='W2')
    b2 = tf.Variable(tf.zeros(shape=n_hidden2), name='b2')
    layer2 = activation(tf.add(tf.matmul(L1_Dat, W2), b2, name='layer2'))

    # --- Layer 3 (hidden)
    W3 = tf.Variable(dtype=tf.float32, initial_value=initializer([n_hidden2, n_hidden3]), name='W3')
    b3 = tf.Variable(tf.zeros(shape=n_hidden3), name='b3')

    # What activation we put here ? Well, the output that we're trying to replicate
    # is that from Layer 1, which goes through a non-linearity.
    # we're comparing directly to the output of the hidden 1 layer !!!
    layer3 = activation(tf.add(tf.matmul(layer2, W3), b3), name='layer3')

    reconstruction_loss = tf.reduce_mean(tf.square(L1_Dat - layer3), name='mse')
    reg_losses = [l2_reg(W2), l2_reg(W3)]
    reg_loss = tf.add_n(reg_losses)
    loss = tf.add_n([reconstruction_loss] + reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session(graph=graph2) as sess:
        sess.run(init)

        # Phase 1 data, is the actual data
        for epoch in range(n_epochs):
            n_batches = len(L1_TrainData) // batch_size
            loss_val = 0.
            for iteration in range(n_batches):
                L1_batch = L1_TrainData[iteration:iteration+batch_size,:]
                sess.run(training_op, feed_dict={L1_Dat: L1_batch})

            print('===== Phase 2 - Epoch: {0} ====='.format(epoch))
            loss_train, reg_loss_train = sess.run([reconstruction_loss, reg_loss], feed_dict={L1_Dat: L1_TrainData})
            loss_val = sess.run(reconstruction_loss, feed_dict={L1_Dat: L1_ValidData})
            print('Train: Reconstruction MSE = {0:.4f}, Regularization: {1:.4f}'.format(loss_train, reg_loss_train))
            print('Validation MSE: {0:.4f}'.format(loss_val))

        L2_Wvals, L2_bvals = W2.eval(), b2.eval()
        L3_Wvals, L3_bvals = W3.eval(), b3.eval()




#################    Plots    #####################
import matplotlib.pyplot as plt

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def show_reconstructed_digits(X, outputs, n_test_digits = 2):
    with tf.Session() as sess:
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
    plt.waitforbuttonpress()

############ Construct the final autoencoder ##############

'''
Constructing the final encoder from the pre-trained components
from the previous phases
'''

# tf.reset_default_graph()
reset_graph()

# Switching back to the default graph here
X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='X')

# -- Layer 1
W1 = tf.constant(value=L1_Wvals, name='W1final')
b1 = tf.constant(value=L1_bvals, name='b1final')
layer1 = activation(tf.matmul(X, W1) + b1, name='layer1')

# --- Layer 2 (codings)
W2 = tf.constant(value=L2_Wvals, name='W2final')
b2 = tf.constant(value=L2_bvals, name='b2final')
layer2 = activation(tf.matmul(layer1, W2) + b2, name='layer2')

# --- Layer 3
W3 = tf.constant(value=L3_Wvals, name='W3final')
b3 = tf.constant(value=L3_bvals, name='b3final')
layer3 = activation(tf.matmul(layer2, W3) + b3, name='layer3')

# --- Output layer
Wout = tf.constant(value=Lout_Wvals, name='Woutfinal')
bout = tf.constant(value=Lout_bvals, name='boutfinal')
output = tf.matmul(layer3, Wout) + bout

# Wout = tf.constant(value=Lout_Wvals, name='Woutfinal')
# bout = tf.constant(value=Lout_bvals, name='boutfinal')
# output = tf.matmul(layer1, Wout) + bout

# Here we only care about reconstruction loss
loss = tf.reduce_mean(tf.square(X - output), name='mse')

# No training here. Just testing
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    test_loss_val = sess.run(loss, feed_dict={X: mnist.test.images})
    valid_loss_val = sess.run(loss, feed_dict={X: mnist.validation.images})
    train_loss_val = sess.run(loss, feed_dict={X: mnist.train.images})

print('========== Final Output ==============')
print('Train: {0:.4f}, Valid: {1:.4f}, Test: {2:.4f} MSE'.format(train_loss_val, valid_loss_val, test_loss_val))

show_reconstructed_digits(X, output)




