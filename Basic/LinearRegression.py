import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing


'''
Manipulates 2D arrays to perform Linear Regression 
on the California housing dataset
'''

housing = fetch_california_housing()
m, n = housing.data.shape
# Add leading bias (constant) to data. This is to account for the
# constant parameter of the linear polynomial expression
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

m,n = housing_data_plus_bias.shape[0], housing_data_plus_bias.shape[1]
print('Samples: {0}, Features: {1}'.format(m, n))
print()


# Set up the data + response as a system of linear equations:
# <y> = X * theta
# where theta is the tensor of the linear polynomial
# Solve the system of linear equations using the pseudo-inverse
# (least squares solution to the )

# https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse
# Inv(X' * X) * X'* <y> = Inv(X' * X) * (X' * X) * theta ==>
# theta = Inv(X' * X) * (X') * <y>
# where Inv(X' * X) * (X') is the Moore–Penrose pseudo inverse,
# Where we get the  lovely Identity Matrix: Inv(X' * X) * (X' * X)] = I


'''
Constants and variables take no input (they are called source ops)
'''
# Create two tensors. One for the data (X), one for the targets
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

'''
operations (also called ops for short) can take any number 
of inputs and produce any number of outputs
'''
# So, nothing is evaluated here.
# create nodes in the graph that will perform them when the graph is run
XT = tf.transpose(X)
# pseudo_inverse
mpinv = tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT)
theta = tf.matmul(mpinv, y)

with tf.Session() as sess:
    theta_value = theta.eval()

# Print the parameters
print('Parameter Values')
print(theta_value)