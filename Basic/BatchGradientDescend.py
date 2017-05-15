import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def run_training_op(training_op, cost_op, n_epochs):
    '''
        Run a training op over a number of epochs
        Assumption: 'theta' variable is expected to be
        in the default graph. (Variables remain "alive"
        for the duration of the session. Ops/graph nodes
        drop their values between graph runs).
    :param training_op: the op to run
    :param n_epochs: number of epochs
    :return: 
    Best paramter 
    '''
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            sess.run(training_op)
            if epoch % 100 == 0:
                print('Epoch: {0}, Cost{1:.2E}'.format(epoch, cost_op.eval()))
                save_path = saver.save(sess, "/tmp/my_model.ckpt")

        # Theta is a variable of the default graph.
        # So a call to it's eval will return the
        # value it has at that point
        best_theta = theta.eval()
        save_path = saver.save(sess, "/tmp/my_model.ckpt")
    return best_theta

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


## Gradient Descend Implementation
n_epochs = 1000
learning_rate = 0.01


###################### Formulaic Method #########################
# Important
# Any node you create is automatically ADDED to the DEFAULT graph

# ----- Source ops ------
# Note the data type declaration. It is tf.float32
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")


# Initialize theta
# Tensor/node in the graph, with shape <n + 1 x 1>, with values in the range -1, 1
randvals = tf.random_uniform([n + 1, 1], -1.0, 1.0, seed = 13) # Run once for comparison
theta = tf.Variable(randvals, name="theta")

# ------ Ops ----------
# Linear equation: y_hat = X*theta
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y


'''
reduce_mean()
    Computes the mean of elements across dimensions of a tensor.
'''
mse = tf.reduce_mean(tf.square(error), name="mse")
# Formula : Batch gradient of the MSE. Scaled by 2/m
gradients = 2/m * tf.matmul(tf.transpose(X), error)

'''
assign() function creates a node that will assign a new value to a variable.
In this case, it assigns the value of (theta - learning_rate * gradients) to
theta... it implements the Batch Gradient Descent step

GT: It's the mathematical "equal" operation. Or,
  theta_j<i+1> <--- theta_j<i> - mu * gradient<theta_j>
  This is used as an operation because we can't use:
  session.run(theta = theta - learning_rate * gradients)
 
Single value
theta_j<i+1> = theta_j<i> - mu * gradient<theta_j>

where the general case, vector form:
theta<i+1> = theta<i> - mu * gradient<theta>
'''
training_op = tf.assign(theta, theta - learning_rate * gradients)

best_theta = run_training_op(training_op=training_op, cost_op=mse, n_epochs=n_epochs)
print('Best Theta found - Manual Gradients')
print(best_theta)


################ Use TF gradient() #####################
# Replace the exact form of gradient, with tf.gradients()

tf.reset_default_graph()

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# Initialize theta
# Tensor/node in the graph, with shape <n + 1 x 1>, with values in the range -1, 1
randvals = tf.random_uniform([n + 1, 1], -1.0, 1.0, seed = 13) # Run once for comparison
theta = tf.Variable(randvals, name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

error = y_pred - y # error_i = y_hat_i - y_i
# (1/m)*sum(error_i^2)
mse = tf.reduce_mean(tf.square(error), name="mse")

'''
The gradients() function takes an op (in this case mse) 
and a list of variables (in this case just theta) and 
it creates a list of ops (one per variable) to compute 
the gradients of the op with regards to each variable. 
So the gradients node will compute the gradient vector of 
the MSE with regards to theta.
'''

# D_MSE / d_theta
# Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`
# Returns: A list of `dth/dx` for each x in `xs`
gradients = tf.gradients(ys = mse, xs = [theta])[0]

# The optimization:  theta(i+i) <--- theta(i) - nu * D_MSE
training_op = tf.assign(theta, theta - learning_rate * gradients)

# Run
best_theta = run_training_op(training_op=training_op, cost_op=mse, n_epochs=n_epochs)

print()
print('Best Theta found - TF Gradients')
print(best_theta)

################# Use GradientDescend Optimizer ###################
tf.reset_default_graph()

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# Initialize theta
# Tensor/node in the graph, with shape <n + 1 x 1>, with values in the range -1, 1
randvals = tf.random_uniform([n + 1, 1], -1.0, 1.0, seed = 13) # Run once for comparison
theta = tf.Variable(randvals, name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

error = y_pred - y # error_i = y_hat_i - y_i
mse = tf.reduce_mean(tf.square(error), name="mse") # (1/m)*sum(error_i^2)

# The optimization:  theta(i+i) = theta(i) - nu * D(MSE)/d_theta
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Find the minima (hopefully the global)
training_op = optimizer.minimize(mse)

# Run
best_theta = run_training_op(training_op=training_op, cost_op=mse, n_epochs=n_epochs)

print()
print('Best Theta found - GradientDescentOptimizer')
print(best_theta)