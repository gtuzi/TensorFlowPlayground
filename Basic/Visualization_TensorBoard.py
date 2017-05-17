import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

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

    :param batch_index: 
    :param batch_size: 
    :return: 
    '''
    # load the data from disk
    i = batch_index * batch_size
    X_batch = scaled_housing_data_plus_bias[i:i + batch_size, :]
    y_batch = target[i:i + batch_size]
    return X_batch, y_batch

# -----------------------------------------------

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
                    epoch_cost += (1./num_batches)*bcost
                    epoch_grad += (1./num_batches)*gradnorm

                    if batch % 10 == 0:
                        # Log MSE data
                        summary_str = mse_summary.eval(feed_dict={X: Xbatch, y: ybatch})
                        step = epoch * n_batches + batch
                        summary_writer.add_summary(summary_str, step)

                        # Log gradient data
                        summary_str = grad_norm_summary.eval(feed_dict={X: Xbatch, y: ybatch})
                        step = epoch * n_batches + batch
                        summary_writer.add_summary(summary_str, step)

                    # print('Cost: {0}'.format(b))
                if epoch % 50 == 0:
                    # Save every 100 epochs
                    save_path = saver.save(sess, "/tmp/my_model.ckpt")
                    print('Epoch: {0}, Epoch Cost: {1:.2E}, Gradient: {2:.2E}'.format(epoch, epoch_cost, epoch_grad))

            save_path = saver.save(sess, "/tmp/my_model.ckpt")
            best_theta = theta.eval()
        return best_theta
    except Exception as ex:
        raise Exception(str(ex))



tf.reset_default_graph()

## Gradient Descend Implementation
n_epochs = 2000
learning_rate = 0.005
batch_size = 1000
n_batches = int(np.ceil(m / batch_size))



########## Graph Construction Phase ####################
# Using placehoders to "feed" data into, instead of variables.
# Note: Shape and type of the data to be held.
#      'None' implies varying size
X = tf.placeholder(dtype=tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

# Initialize theta
# Tensor/node in the graph, with shape <n + 1 x 1>, with values in the range -1, 1
randvals = tf.random_uniform([n + 1, 1], -1.0, 1.0, seed = 13) # Run once for comparison
theta = tf.Variable(randvals, name="theta")
y_pred = tf.matmul(X, theta, name="predictions")


with tf.name_scope("loss") as scope:
    error = y_pred - y # error_i = y_hat_i - y_i
    mse = tf.reduce_mean(tf.square(error), name="mse") # (1/m)*sum(error_i^2)

gradients = tf.gradients(mse, theta)[0]

# Keep track of the gradient norm. If we have improvements, the gradients
# must get smaller - if theta is in the optimal valley
gradient_norm_op = tf.sqrt(tf.reduce_sum(tf.square(gradients)))

# The optimization:  theta(i+i) <--- theta(i) - nu * D_MSE
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
saver = tf.train.Saver()



'''
Use a different log directory every time you run your 
program, or else TensorBoard will merge stats from different runs
'''
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# will evaluate the MSE value and write it to a TensorBoard-compatible binary log
# string called a summary
mse_summary = tf.scalar_summary('MSE', mse)
grad_norm_summary = tf.scalar_summary('Gradient_Norm', gradient_norm_op)

# Write summaries to log files in the log directory
# - first parameter indicates the path of the log directory
# - second (optional) parameter is the graph you want to visualize
summary_writer = tf.train.SummaryWriter(logdir, tf.get_default_graph())

###### End of construction phase ##########


####### Graph Execution Phase #############
best_theta = run_training_op(training_op=training_op,
                             n_epochs=n_epochs,
                             batch_size=batch_size,
                             num_batches=n_batches,
                             cost_op=mse,
                             init=init,
                             saver=saver)

# Must be called to be closed
summary_writer.close()
