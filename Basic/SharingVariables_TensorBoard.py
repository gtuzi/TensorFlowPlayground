import tensorflow as tf

'''
How to share variables across reused code
'''

# Option 1: pass the (tf) variable as an argument
def relu1(X, threshold):
    '''
    Construct and return a RELU function
    :param X: Input data
    :param threshold: Relu threshold value
    :return: 
    Linearly rectified output
    '''
    '''
        Create name scopes to group related nodes.
        The name_scope will put a decorative box
        around the items (variables/functions/constants)
        under its scope.
    '''
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="relu")


# Option 2: Set the shared variable as an attribute of the relu() function
#           upon the first call
def relu2(X, threshold):
    '''
    Construct and return a RELU function
    :param X: Input data
    :param threshold: Relu threshold value
    :return: 
    Linearly rectified output
    '''
    '''
        Create name scopes to visually group related nodes (in TensorBoard)
        The name_scope will put a decorative box
        around the items (variables/functions/constants)
        under its scope.
    '''
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")

        if hasattr(relu2, 'threshold'):
            thresh = relu2.threshold
        else:
            thresh = 0.

        return tf.maximum(z, thresh, name="relu")


# -----------------------------------------------------
# Option 3: use the get_variable() to reuse shared variable
#           it if it already exists.
#           Create shared variable OUTSIDE the consumer function
def relu3(X):
    '''
    Construct and return a RELU function
    :param X: Input data
    :param threshold: Relu threshold value
    :return: 
    Linearly rectified output
    '''
    '''
        Create name scopes to visually group related nodes (in TensorBoard)
        The name_scope will put a decorative box
        around the items (variables/functions/constants)
        under its scope.
    '''
    with tf.name_scope("relu3"):

        # Create/reuse a variable named 'relu/threshold'
        # If you want to reuse a variable, you need to
        # explicitly say so by setting the variable
        # scope’s reuse attribute to True
        with tf.variable_scope("shared_relu_var", reuse=True) as var_scope:
            try:
                # Reuse 'relu/threshold'
                # This code will fetch the existing "relu/threshold" variable. This
                # variable can be set inside this function, or outside somewhere else.
                # (or raise an exception if it does not exist or if it
                # was not created using get_variable()).
                # So this is practically an input to this reusable function
                thresh = tf.get_variable('threshold')
            except Exception as ex:
                print('Shared variable not found: {0}'.format(str(ex)))
                raise Exception(ex)


        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")

        return tf.maximum(z, thresh, name="relu")


# Opt 3 - Shared variable creation
try:
    with tf.variable_scope('shared_relu_var'):
        thresh = tf.get_variable('threshold', shape=(),
                                 initializer=tf.constant_initializer(0., dtype=tf.float32))
except:
    # if the variable has already been created by an earlier
    # call to get_vari able(), then this code will raise an exception.
    pass
n_features = 9
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu3(X) for i in range(4)]

# Add the input from multiple Relu's
output = tf.add_n(relus, name="output")


# ------------------------------------------------------
# Option 4: Create INSIDE the consumer function, but flag in client code
#           creates the threshold variable within the relu()
#           function upon the first call, then reuses it in subsequent calls.
def relu4(X):
    '''
    Construct and return a RELU function
    :param X: Input data
    :param threshold: Relu threshold value
    :return: 
    Linearly rectified output
    '''
    '''
        Create name scopes to visually group related nodes (in TensorBoard)
        The name_scope will put a decorative box
        around the items (variables/functions/constants)
        under its scope.
    '''
    with tf.name_scope("relu4"):
        # Create/reuse a variable named 'relu/threshold'
        thresh = tf.get_variable('threshold', shape=(),
                                 initializer=tf.constant_initializer(0., dtype=tf.float32))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")

        return tf.maximum(z, thresh, name="max_relu")


# # Option 4 - Shared variable creation code
# n_features = 9
# X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
#
# relus = []
# for relu_index in range(5):
#     # Note the reuse becomes True after the first creation
#     # Also, the relu function created and enlisted, fall under the scope
#     # of the 'shared_relu_var'. So the list below, in TensorBoard, will create
#     # a sequence of 'shared_relu_var_x' with the 'relu4' function/name_scope in them
#     with tf.variable_scope("shared_relu_var", reuse=(relu_index >= 1)) as scope:
#         relus.append(relu4(X))
#         output = tf.add_n(relus, name="output")


# ----------------------------------------------------- #
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Write summaries to log files in the log directory
# - First:  indicates the path of the log directory
# - Second: (optional) parameter is the graph you want to visualize
#           Recommend: always put this guy in !!
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Must be called
summary_writer.close()