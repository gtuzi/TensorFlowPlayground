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

# ----- Constants  -------
# Investigate shape and rank
# Rank = 0
constLit = tf.constant(4.5, shape=()) # Still a tf.op.Tensor
# Rank = 1: [3.5, 5.6]
constA = tf.constant([3.5, 5.6], shape=(2,))
# Rank = 2: [[3.5, 5.6]]
constB = tf.constant([3.5, 5.6], shape=(1,2))
# Rank = 2: [[3.5,
#             5.6]]
constC = tf.constant([3.5, 5.6], shape=(2,1))

constMa = tf.constant([[1, 2], [9, 4]], shape=(2,2), dtype=tf.float32)
constMb = tf.constant([[5, 6], [7, 8]], shape=(2,2), dtype=tf.float32)


x = tf.Variable(xpy, dtype=tf.float32, name="x")
y = tf.Variable(ypy, dtype=tf.float32, name="y")

M = tf.concat(concat_dim=0, values=[x, y])

# Elementwise squaring
M2 = tf.square(M)

# Get the shape of a tensor as a list of two integers
shp = M.get_shape().as_list()

# Division
coeff_op = tf.div(tf.constant(1., dtype=tf.float32),
                  tf.constant(shp[0], dtype=tf.float32))

# Sum along a particular axis
sum_op = tf.reduce_sum(M2, axis=0)

# Multiplication
favg_man = tf.mul(coeff_op, sum_op)

# Get the mean across a particular axis of a 2+ dimension tensor
favg_tf = tf.reduce_mean(M2, axis=0)

# Add (addition) item by item
xyAddN = tf.add_n([x, y])


# Losses
dM = constMa - constMb
mse_loss = tf.contrib.losses.mean_squared_error(constMa, constMb)


####### Inverse Convolution ###############
batch_size = 5
input_channels = 128
output_channels = 96
output_shape = [batch_size, 100, 100, output_channels]
input_shape = [batch_size, 50, 50, input_channels]

strides = [1, 2, 2, 1]

inputSignal = tf.constant(0.1, shape=input_shape)
outputSignal = tf.constant(0.5, shape=output_shape)

W = tf.constant(0.1, shape=[4, 4, output_channels, input_channels])

forwardConv = tf.nn.conv2d(outputSignal, W, strides=strides, padding='SAME')

invConvOut = tf.nn.conv2d_transpose(inputSignal, W,
                                 output_shape=output_shape,
                                 strides=strides,
                                 padding='SAME')

init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize ALL the variables (variable.initializer.run())
    print('------- Variables, Tensors and Constants --------')
    print('constLit: {0}, \tshape: {1}, rank: {2}'.format(constLit.eval(), constLit.get_shape().as_list(), tf.rank(constLit).eval()))
    print('constA: {0}, \tshape: {1}, rank: {2}'.format(constA.eval(), constA.get_shape().as_list(), tf.rank(constA).eval()))
    print('constB: {0}, \tshape: {1}, rank: {2}'.format(constB.eval(), constB.get_shape().as_list(), tf.rank(constB).eval()))
    print('constC: {0}, \tshape: {1}, rank: {2}'.format(constC.eval(), constC.get_shape().as_list(),tf.rank(constC).eval()))


    print('\n------- Operations ------- ')
    print('Coeff_op: {0}'.format(coeff_op.eval()))
    print('Shape of x: {0}'.format(tf.shape(x).eval()))
    print('M: {0}'.format(M.eval()))
    print('M^2: {0}'.format(M2.eval()))
    print('Sum_op: {0}'.format(sum_op.eval()))
    print('Favg_man: {0}'.format(favg_man.eval()))
    print('Favg_tf: {0}'.format(favg_tf.eval()))
    print('XYAddN: {0}'.format(xyAddN.eval()))

    print('\n------- Losses ------- ')
    print('Delta Matrix {0}'.format(dM.eval()))
    print('MSE: {0}'.format(mse_loss.eval()))

    inputSignalVal = inputSignal.eval()
    outputSignalVal = outputSignal.eval()

    print('\n------- Forward Convolution ------- ')
    forwardConvVal = forwardConv.eval()
    print('Forward Conv Shape: {0}'.format(forwardConvVal.shape))
    print('\n------- Inverse Convolution ------- ')
    invConvOutVal = invConvOut.eval()
    print('Inv Conv Shape: {}'.format(invConvOutVal.shape))
    pass




