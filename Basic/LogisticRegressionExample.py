'''
    Implement the training pipeline for logistic regression,
    using mini-batch learning.
'''
import os
import tensorflow as tf

from sklearn.datasets import make_moons, load_iris
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support

################ Data #####################

iris = load_iris()
print(iris['feature_names'])

# # Petal Length (3 - 7), Petal Width (0 - 3)
X_orig = iris["data"][:, 2:]
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0


# n_samples = 1000
# X_orig, y = make_moons(n_samples=n_samples, noise=0.1, shuffle=True)


# Add the constant term
X = np.c_[np.ones((X_orig.shape[0], 1)), X_orig]
(n_samples,nfeats) = X.shape

# Standard scale the input data
scaler = StandardScaler()
X = scaler.fit_transform(X=X)


def fetch_batch(batch_size):
    '''
        Extract batch data. Shiffles the data at the beginning 
        of every epoch.
    :param batch_index: batch index within the epoch 
    :param batch_size: number of samples in a batch
    :return: 
    (X-batch, y-batch)
    '''
    global X
    global y
    n_samps = len(y)

    while 1:
        X_, y_ = shuffle(X, y)
        for i in range(0, n_samps, batch_size):
            X_batch = X_[i:i + batch_size, :]
            y_batch = y_[i:i + batch_size]
            yield X_batch, y_batch

################  Graph Construction ################

def logistic_regression_construct(X, W):
    '''
        Construct logistic regression function (op). Compute the logit (prob), logprob
    :param X: input data tensor with shape (nsamples, nfeatures)
    :return: logit(prob)
    '''
    with tf.name_scope('logistic'):
        lin = tf.matmul(X, W, name='lin')
        logit = tf.div(tf.constant(1., dtype=tf.float32),
                       tf.add(tf.constant(1., dtype=tf.float32), tf.exp(-lin)), name='logit')

    return logit


def logit_pred(logprob):
    '''
    Use the output of the logistic regression to generate binary classes as:
    1 - logprob >= 0.5
    0 - otherwise
    :param logprob: log probability
    :return: 
    '''

    with tf.name_scope('logit_pred'):
        thresh = 0.5
        # logit >= thresh
        return tf.select(tf.greater_equal(logprob, thresh),
                         tf.ones_like(logit), # If true
                         tf.zeros_like(logit), name='pred')

tf.reset_default_graph()

LR = 0.001
batch_size = 20
num_batches = np.ceil(X.shape[0] / float(batch_size))
num_epochs = 5000

# --- Source Ops
# The None element of the shape corresponds to a variable-sized dimension
X_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='X')
y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,], name='y')
w_shape = (nfeats, 1)
W = tf.Variable(tf.random_normal(shape=w_shape), name='W')

# --- Logistic Op
logit = logistic_regression_construct(X_placeholder, W)

# --- Decision Operation
pred_op = logit_pred(logit)

# --- Cost Op
# cost function is the mean of binary cross-entropy
with tf.name_scope('cost'):
    posentropy = tf.mul(y_placeholder, tf.log(logit), name='positive_entropy')  # y = 1
    negentropy = tf.mul((1 - y_placeholder), tf.log(1 - logit), name='negative_entropy')  # y = 0
    cost = -tf.reduce_mean(posentropy + negentropy, name='cost') # Mean cost

with tf.name_scope('gradient'):
    # --- Cost Gradient: dMean_Cost/dw
    cost_grad = tf.gradients(cost, [W], name='gradient')[0]
    # --- Cost Gradient Norm:
    gradient_norm_op = tf.sqrt(tf.reduce_sum(tf.square(cost_grad)),'gradient_norm')

with tf.name_scope('training'):
    # Batch gradient descend: for a given batch
    # W(t+1) <--- W(t) -  LR * grad(cost)/W(t)
    training = tf.assign(W, W - LR * cost_grad, name='training')



################### Graph Execution ###################
'''
Implement Mini-batch Gradient Descent. For this, we need a way to 
replace X and y at every iteration with the next mini-batch. 
The simplest way to do this is to use placeholder nodes.
These nodes are special because they don’t actually perform any 
computation, they just OUTPUT the data you tell them to output at RUNTIME. 
They are typically used to pass the training data to TensorFlow during training. 
If you don’t specify a value at runtime for a placeholder, you get an exception.
'''

def scores(y, y_pred):
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true=y,y_pred=y_pred, average='binary')
    return precision, recall, f_score

def run_training_op(training_op, cost_op, pred_op, n_epochs):
    '''
        Implement batch training
        Run a training op over a number of epochs.
        Variables assumed to be in the default graph: 'theta'
    :param batch_size : number of samples per batch
    :param num_batches: number of batches per epoch
    :param training_op: Training operation to run 
    :param pred_op: prediction operation
    :param n_epochs: number of epochs
    :return: 
        Best weight (W)
    '''

    global init
    global saver
    global batch_size
    global merged
    ####### Graph Execution Phase #############
    try:

        genfun = fetch_batch(batch_size)
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                epoch_cost = 0.
                epoch_grad_norm = 0.
                epoch_recall = 0.
                epoch_precision = 0.
                epoch_f1 = 0.
                for batch in range(int(num_batches)):
                    Xbatch, ybatch = next(genfun)
                    # Run Gradients and then add
                    _, gradnorm, bcost, y_pred = sess.run([training_op, gradient_norm_op, cost_op, pred_op],
                                                  feed_dict={X_placeholder: Xbatch, y_placeholder: ybatch})

                    precision,recall,f1 = scores(ybatch, y_pred.squeeze().astype(np.int32))
                    epoch_cost += (1./num_batches)*bcost
                    epoch_grad_norm += (1./num_batches)*gradnorm
                    epoch_recall += (1./num_batches)*recall
                    epoch_precision += (1./num_batches)*precision
                    epoch_f1 +=(1./num_batches)*f1
                if epoch % 50 == 0:
                    # Save every 50 epochs
                    save_path = saver.save(sess, "/tmp/my_model.ckpt")

                    step = epoch * num_batches + batch
                    summary_str = sess.run(merged, feed_dict={X_placeholder: Xbatch,
                                                              y_placeholder: ybatch,
                                                              recall_placeholder:epoch_recall,
                                                              precision_placeholder:epoch_precision,
                                                              f1_placeholder:epoch_f1})
                    summary_writer.add_summary(summary_str, step)
                    print(str('Epoch: {0}, Cost: {1:.4E}, ' +
                              'Gradient Norm: {2:.4E}, Recall: {3:.2f},' +
                              ' Precision: {4:.2f}, F1: {5:.2f}').format(epoch, epoch_cost,
                                                                         epoch_grad_norm,
                                                                         epoch_recall,
                                                                         epoch_precision,
                                                                         epoch_f1))

            save_path = saver.save(sess, "/tmp/my_model.ckpt")
            best_W = W.eval()
        return best_W
    except Exception as ex:
        raise Exception(str(ex))


init = tf.global_variables_initializer()

# Saver saves the variables of the session
saver = tf.train.Saver()


# Score holders
recall_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='recall_tensor')
recall_summary = tf.summary.scalar('Recall', recall_placeholder)
precision_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='recall_tensor')
precision_summary = tf.summary.scalar('Precision', precision_placeholder)
f1_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='f1_tensor')
f1_summary = tf.summary.scalar('F1', f1_placeholder)

# output_summary = tf.scalar_summary('Output', pred_op)
pred_summary = tf.summary.tensor_summary('Prediction', pred_op)
# cost_summary = tf.scalar_summary('Cost', cost)
cost_summary = tf.summary.scalar('Cost', cost)
# gradient_summary = tf.scalar_summary('Gradient', cost_grad)
graident_summary = tf.summary.tensor_summary('Gradient', cost_grad)
# gradient_norm_summary = tf.scalar_summary('GradientNorm', gradient_norm_op)
grad_norm_summary = tf.summary.scalar('Gradient Norm', gradient_norm_op)

################# Save Logs ##################
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-LogisticRegression{}/".format(root_logdir, now)


# Write summaries to log files in the log directory
# - First:  indicates the path of the log directory
# - Second: (optional) parameter is the graph you want to visualize
#           Recommend: always put this guy in !!
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


##################### Show Time ###############
best_W = run_training_op(training_op=training, cost_op=cost, pred_op= pred_op, n_epochs=num_epochs)
print('')
print(best_W)

# Must be called at the end
summary_writer.close()


#################### Visualize ####################
import matplotlib.pyplot as plt

tf.reset_default_graph()


# Add variables to the graph
X_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='X')
y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,], name='y')
# w_shape = (nfeats, 1)
W = tf.Variable(best_W, name='W')

# --- Ops
# Compute the logit (prob), logprob,
# Add opp to graph
logit = logistic_regression_construct(X_placeholder, W)
# --- Decision Operation
pred_op = logit_pred(logit)
init = tf.global_variables_initializer()


# --- Using the learned weights, visualize the decision space
sess = tf.InteractiveSession()
sess.run(init)
prob = sess.run(logit, feed_dict={X_placeholder:X})
# logit, logprob = logistic_regression_construct(X_placeholder, W)
# pred_op = logit_pred(logit)
sess.close()

pos_idc = (prob >= 0.5).squeeze()
neg_idc = (prob < 0.5).squeeze()


plt.subplot(2, 1, 1)
plt.scatter(X_orig[pos_idc, 0], X_orig[pos_idc, 1], color="g",  marker='^')
plt.scatter(X_orig[neg_idc, 0], X_orig[neg_idc, 1], color="r", marker='s')
plt.title('Predicted Classes')

plt.subplot(2, 1, 2)
plt.scatter(X_orig[(y >= 1).squeeze(),0], X_orig[(y >= 1).squeeze(),1], color='g', marker='^')
plt.scatter(X_orig[(y <= 0).squeeze(),0], X_orig[(y <= 0).squeeze(),1], color='r', marker='s')
plt.title('True Classes')
plt.waitforbuttonpress()
pass
