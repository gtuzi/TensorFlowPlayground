'''
    Use one graph and two phases to train a { Input -- Hidden 1 -- Coding -- Hidden 3 -- Output }
    autoencoder network
'''

import tensorflow as tf
import numpy as np

################## Load Data ###############
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")


#################    Plots    #####################
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

##################  Parameters ##############
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



n_inputs = 28 * 28 # for MNIST
n_hidden1 = 300
n_hidden2 = 150 # codings
n_hidden3 = n_hidden1
n_output = n_inputs
learning_rate = 0.01
l2_reg = 0.0001

n_epochs = 4
batch_size = 150

reset_graph()

activation = tf.nn.elu
l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='X')

# ---------- Weights -----------
# --- Layer 1 (Hidden)
W1 = tf.Variable(dtype=tf.float32, initial_value=initializer([n_inputs, n_hidden1]), name='W1')
b1 = tf.Variable(tf.zeros(shape=n_hidden1), name='b1')

# ---- Coding layer
W2 = tf.Variable(dtype=tf.float32, initial_value=initializer([n_hidden1, n_hidden2]), name='W2')
b2 = tf.Variable(tf.zeros(shape=n_hidden2), name='b2')

# --- Layer 3 (hidden)
W3 = tf.Variable(dtype=tf.float32, initial_value=initializer([n_hidden2, n_hidden3]), name='W3')
b3 = tf.Variable(tf.zeros(shape=n_hidden3), name='b3')

# --- Output layer
Wout = tf.Variable(dtype=tf.float32, initial_value=initializer([n_hidden3, n_output]), name='Wout')
bout = tf.Variable(tf.zeros(shape=n_output), name='bout')

hidden1 = activation(tf.matmul(X, W1) + b1, name='hidden1')
hidden2 = activation(tf.matmul(hidden1, W2) + b2, name='hidden2')
hidden3 = activation(tf.matmul(hidden2, W3) + b3, name='hidden3')
output = tf.add(tf.matmul(hidden3, Wout), bout, name='output')

reconstruction_loss = tf.reduce_mean(tf.square(X - output), name='mse')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


with tf.name_scope('phase1'):
    ph1_outputs = tf.matmul(hidden1, Wout) + bout
    ph1_reg_loss = [l2_reg(W1), l2_reg(Wout)]
    ph1_reconstruction_loss = tf.reduce_mean(tf.square(X - ph1_outputs))
    ph1_loss = tf.add_n([ph1_reconstruction_loss] + ph1_reg_loss)
    ph1_training_op = optimizer.minimize(ph1_loss)

with tf.name_scope('phase2'):
    ph2_reg_loss = [l2_reg(W2), l2_reg(W3)]
    # This will cause the raw input to still go through layer 1.
    # a much faster approach is to pre-generate layer 1 output
    ph2_reconstruction_loss = tf.reduce_mean(tf.square(hidden1 - hidden3))
    ph2_loss = tf.add_n([ph2_reconstruction_loss] + ph2_reg_loss)
    # Freeze W1 and Wout
    train_vars = [W2, b2, W3, b3]
    ph2_training_op = optimizer.minimize(ph2_loss, var_list=train_vars)

training_ops = [ph1_training_op, ph2_training_op]
reconstruction_losses = [ph1_reconstruction_loss, ph2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

init = tf.global_variables_initializer()
# saver = tf.train.Saver()

W1_val = None
W2_val = None
W3_val = None
Wout_val = None

with tf.Session() as sess:
    sess.run(init)
    i = 0
    for t_op, recon_loss, batch_size, n_epoch in zip(training_ops, reconstruction_losses, batch_sizes,n_epochs):
        i +=1
        print('Phase {0}'.format(i))

        for epoch in range(n_epoch):
            print('Epoch: {0}'.format(epoch))
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(t_op, feed_dict={X: X_batch})

            loss_train = recon_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            # saver.save(sess, "./my_model_one_at_a_time.ckpt")

        loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
        print("Test MSE:", loss_test)

    # Capture the weight values for visualiztions
    W1_val = W1.eval()
    W2_val = W2.eval()
    W3_val = W3.eval()
    Wout_val = Wout.eval()

    show_reconstructed_digits(X, output)

figure = plt.figure()
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plot_image(W1_val.T[i])
plt.waitforbuttonpress()

# save_fig("extracted_features_plot") # not shown
plt.show()                          # not shown
