import tensorflow as tf

'''
    Visualize and modularize a graph
'''



def relu(X):
    '''
    Construct and return a RELU function
    :param X: Input data
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
        return tf.maximum(z, 0., name="relu")


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]


# Adds all input tensors element-wise
output = tf.add_n(relus, name="output")

output_summary = tf.scalar_summary('Output', output)


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
