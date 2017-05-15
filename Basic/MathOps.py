import tensorflow as tf
import numpy as np

# ----- Vector Operations
## Note: no computation is performed here
# Declare variables, and give them a value
# The variables have not been initialized

# Initialize vector tensors. Note that they you need as many
# [] as the order of the tensor. The number of '[[' equates
# to the order of tensor
xpy = [[3, 5, 6]]
ypy = [[1, 2, 3]]
x = tf.Variable(xpy, dtype=tf.float32, name="x")
y = tf.Variable(ypy, dtype=tf.float32, name="y")
M = tf.concat(concat_dim=0, values=[x, y])
M2 = tf.square(M)

coeff_op = tf.div(tf.constant(1., dtype=tf.float32), tf.constant(tf.shape(M)[0].scalar(), dtype=tf.float32))
sum_op = tf.reduce_sum(M2, axis=0)
favg_man = tf.mul(coeff_op, sum_op)
favg_tf = tf.reduce_mean(M2, axis=0)

init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize ALL the variables (variable.initializer.run())

    print('Coeff_op: {0}'.format(coeff_op.eval()))
    print('Shape of x: {0}'.format(tf.shape(x).eval()))
    print('M: {0}'.format(M.eval()))
    print('M^2: {0}'.format(M2.eval()))
    print('Sum_op: {0}'.format(sum_op.eval()))
    print('Favg_man: {0}'.format(favg_man.eval()))
    print('Favg_tf: {0}'.format(favg_tf.eval()))



