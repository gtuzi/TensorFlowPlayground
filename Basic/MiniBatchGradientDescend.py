import tensorflow as tf
from sklearn.utils import shuffle

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

####### Placeholder example ###########
# A = tf.placeholder(tf.float32, shape=(None, 3))
# B=A+5
#
# with tf.Session() as sess:
#     # B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
#     B_val_1 = sess.run(B, feed_dict={A:[[1, 2, 3]]})
#     B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
#
# print('B_val_1: {0}'.format(B_val_1))
# print('B_val_2: {0}'.format(B_val_2))

'''
You can actually feed the output of any operations, 
not just place‐holders. In this case TensorFlow does 
not try to evaluate these operations, it uses the values you feed it.
'''

#################### Mini-Batch Gradient Descent ##################
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# global scaled_housing_data_plus_bias
# global target
################ Data Load #################
housing = fetch_california_housing()
m, n = housing.data.shape
# Add leading bias (constant) to data. This is to account for the
# constant parameter of the linear polynomial expression
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data].astype(np.float32)
target = housing.target.reshape(-1, 1).astype(np.float32)
print('Samples: {0}, Features: {1}'.format(m, n))
print()

# Scale data to speed up gradient descend
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)


def fetch_batch(batch_index, batch_size):
    '''
        Extract batch data. Shiffles the data at the beginning 
        of every epoch.
    :param batch_index: batch index within the epoch 
    :param batch_size: number of samples in a batch
    :return: 
    (X-batch, y-batch)
    '''
    global scaled_housing_data_plus_bias
    global target

    if batch_index == 0:
        scaled_housing_data_plus_bias, target = \
            shuffle(scaled_housing_data_plus_bias, target)


    i = batch_index*batch_size
    X_batch = scaled_housing_data_plus_bias[i:i + batch_size, :]
    y_batch = target[i:i + batch_size]
    return X_batch, y_batch

# ------------------------------------------------------------ #
'''
Implement Mini-batch Gradient Descent. For this, we need a way to 
replace X and y at every iteration with the next mini-batch. 
The simplest way to do this is to use placeholder nodes.
These nodes are special because they don’t actually perform any 
computation, they just OUTPUT the data you tell them to output at RUNTIME. 
They are typically used to pass the training data to TensorFlow during training. 
If you don’t specify a value at runtime for a placeholder, you get an exception.
'''
#################  GradientDescend Optimizer ###################

def eval_function(function_ops, n_epochs, batch_size, num_batches):
    '''
        Evaluate function at a given value
    :param function_ops: 
    :param n_epochs: 
    :param batch_size: 
    :param num_batches: 
    :return: 
    '''
    try:
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                for batch in range(num_batches):
                    fun_vals = []
                    Xbatch, ybatch = fetch_batch(batch_index=batch, batch_size=batch_size)
                    for op in function_ops:
                        val = sess.run(op, feed_dict={X: Xbatch, y:ybatch})
                        fun_vals.append(val)

                # Print the first batch sample values
                print('')
                for fval in fun_vals:
                    print(fval.squeeze().tolist())
                    print('-------------------')
    except Exception as ex:
        raise Exception(str(ex))

def eval_gradient_op(grad_ops, n_epochs, batch_size, num_batches):

    '''
        Evaluate gradient operations for the graph at a given value. 
        Used as a sanity check to make sure that 
        the varying gradient methods are consistent with each other
    :param grad_ops: list of gradient operations
    :param n_epochs: number of epochs
    :param batch_size: self explanatory
    :param num_batches: self explanatory
    :return: nothing
    '''

    try:
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                for batch in range(num_batches):
                    grad_vals = []
                    Xbatch, ybatch = fetch_batch(batch_index=batch, batch_size=batch_size)
                    for op in grad_ops:
                        val = sess.run(op, feed_dict={X: Xbatch, y:ybatch})
                        grad_vals.append(val)

                # Print the first batch sample values
                print('')
                for gval in grad_vals:
                    print(gval.squeeze().tolist())
                    print('-------------------')
    except Exception as ex:
        raise Exception(str(ex))


def run_training_op(training_op, cost_op, n_epochs, batch_size, num_batches, init, saver):
    '''
        Implement batch training
        Run a training op over a number of epochs.
        Variables assumed to be in the default graph: 'theta'
    :param batch_size : number of samples per batch
    :param num_batches: number of batches per epoch
    :param training_op: Training operation to run 
    :param n_epochs: number of epochs
    :return: 
        Best parameter (theta)
    '''

    ####### Graph Execution Phase #############
    try:
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                epoch_cost = 0.
                epoch_grad = 0.
                for batch in range(num_batches):
                    Xbatch, ybatch = fetch_batch(batch_index=batch, batch_size=batch_size)
                    # Run Gradients and then add
                    _, gradnorm, bcost = sess.run([training_op, gradient_norm_op, cost_op], feed_dict={X: Xbatch, y: ybatch})
                    epoch_cost += bcost
                    epoch_grad += gradnorm

                    # print('Cost: {0}'.format(b))
                if epoch % 50 == 0:
                    # Save every 100 epochs
                    save_path = saver.save(sess, "/tmp/my_model.ckpt")
                    print('Epoch: {0}, Cost: {1:.2E}, Gradient: {2:.2E}'.format(epoch, epoch_cost, epoch_grad))

            save_path = saver.save(sess, "/tmp/my_model.ckpt")
            best_theta = theta.eval()
        return best_theta
    except Exception as ex:
        raise Exception(str(ex))


tf.reset_default_graph()

## Gradient Descend Implementation
n_epochs = 1000
learning_rate = 0.01
batch_size = 500
n_batches = int(np.ceil(m / batch_size))


########## Graph Construction Phase ####################
# Using placehoders to "feed" data into, instead of variables.
# Note: Shape and type of the data to be held.
#      'None' implies varying size
X = tf.placeholder(dtype=tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")


# Initialize theta
# Tensor/node in the graph, with shape <n + 1 x 1>, with values in the range -1, 1
randvals = tf.random_uniform([n + 1, 1], -1.0, 1.0) # Run once for comparison
theta = tf.Variable(randvals, name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y # error_i = y_hat_i - y_i

# Careful: Computation of MSE involves a division by the number of
#          samples in the batch.
# (1/batch_size)*sum(error_i^2)
mse=tf.reduce_mean(tf.square(error), name="mse")

# ---- Formulaic Method
# Note: divides by batch size (not m: the size of the whole dataset)
# Formula : Batch gradient of the MSE.
#           MSEgradient_theta: (2/batch_size)* (X)' * (error)
gradients = (2/batch_size * tf.matmul(tf.transpose(X), error))

# ---- Using TF gradients()
# D_MSE / d_theta
# Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`
#   Returns: A list of `dth/dx` for each x in `xs`
gradients = tf.gradients(mse, theta)[0]

# Keep track of the gradient norm. If we have improvements, the gradients
# must get smaller - if theta is in the optimal valley
gradient_norm_op = tf.sqrt(tf.reduce_sum(tf.square(gradients)))


# The optimization:  theta(i+i) <--- theta(i) - nu * D_MSE
training_op = tf.assign(theta, theta - learning_rate * gradients)

# ----- Using TF Optimizer
# # The optimization:  theta(i+i) = theta(i) - nu * D(MSE)/d_theta
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()
saver = tf.train.Saver()
###### End of construction phase ##########


####### Graph Execution Phase #############
# eval_function([mse, mse1], n_epochs=n_epochs, batch_size=batch_size, num_batches=n_batches)
# eval_gradient_op(gradient_ops, n_epochs=n_epochs, batch_size=batch_size, num_batches=n_batches)
best_theta = run_training_op(training_op=training_op,
                             cost_op=mse,
                             n_epochs=n_epochs,
                             batch_size=batch_size,
                             num_batches=n_batches,
                             init =init,
                             saver=saver)


# -------- Rresults -------------
print()
print('Best Theta found - GradientDescentOptimizer')
print(best_theta)
