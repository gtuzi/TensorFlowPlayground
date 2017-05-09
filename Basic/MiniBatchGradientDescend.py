import tensorflow as tf


################## Place Holders Example #################
'''
placeholder nodes. 
These nodes are special because they don’t 
actually perform any computation, they just 
output the data you tell them to output at 
runtime. They are typically used to pass the 
training data to TensorFlow during training.

you can also specify its shape, if you want to
 enforce it. If you specify None for a dimension, 
 it means “any size”. 
'''

A = tf.placeholder(tf.float32, shape=(None, 3))
B=A+5

with tf.Session() as sess:
    # B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_1 = sess.run(B, feed_dict={A:[[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print('B_val_1: {0}'.format(B_val_1))
print('B_val_2: {0}'.format(B_val_2))

'''
You can actually feed the output of any operations, 
not just place‐ holders. In this case TensorFlow does 
not try to evaluate these operations, it uses the values you feed it.
'''

#################### Mini-Batch Gradient Descent ##################
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

################ Data Load #################
housing = fetch_california_housing()
m, n = housing.data.shape
# Add leading bias (constant) to data. This is to account for the
# constant parameter of the linear polynomial expression
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

print('Samples: {0}, Features: {1}'.format(m, n))
print()

# Scale data to speed up gradient descend
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)


################# Use GradientDescend Optimizer ###################

def run_training_op(training_op, cost_op, n_epochs):
    '''
        Run a training op over a number of epochs
    :param training_op: 
    :param n_epochs: 
    :return: 
    '''
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "Cost =", cost_op.eval())
            sess.run(training_op)

        best_theta = theta.eval()
    return best_theta


tf.reset_default_graph()

## Gradient Descend Implementation
n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

X = tf.placeholder(dtype=tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

# Initialize theta
# Tensor/node in the graph, with shape <n + 1 x 1>, with values in the range -1, 1
randvals = tf.random_uniform([n + 1, 1], -1.0, 1.0, seed = 13) # Run once for comparison
theta = tf.Variable(randvals, name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

error = y_pred - y # error_i = y_hat_i - y_i
mse = tf.reduce_mean(tf.square(error), name="mse") # (1/m)*sum(error_i^2)

# The optimization:  theta(i+i) = theta(i) - nu * D(MSE)/d_theta
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Run
best_theta = run_training_op(training_op=training_op, cost_op=mse, n_epochs=n_epochs)

print()
print('Best Theta found - GradientDescentOptimizer')
print(best_theta)
