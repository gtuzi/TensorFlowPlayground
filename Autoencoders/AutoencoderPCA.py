'''
simple linear Autoencoder to perform PCA on a 3D dataset,
projecting it to 2D
'''

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import numpy.random as rnd

############### Load Data ####################
rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

####################  Construct Graph     ############################
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

n_inputs = 3 # 3D inputs
n_hidden = 2 # 2D codings

# For autoencoders, number of outputs is
# the same as that of the inputs
n_outputs = n_inputs

learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

# Performing PCA : activation functions are linear (None)
hidden = fully_connected(X, n_hidden, activation_fn=None)
outputs = fully_connected(hidden, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

###################  Execute Graph  ######################
n_iterations = 4000
codings = hidden # the output of the hidden layer provides the codings
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train}) # no labels (unsupervised)
        [codings_val, loss] = sess.run([codings, reconstruction_loss], feed_dict={X: X_test})
        # codings_val = codings.eval(feed_dict={X: X_test})
        if (iteration % 100) == 0:
            print('Iteration: {0}, Loss: {1:.3f}'.format(iteration, loss))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(211, projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], X_test[:,2], marker='o')

plt.subplot(212)
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.tight_layout()
plt.show()