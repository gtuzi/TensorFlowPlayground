import tensorflow as tf


######### Lifecycle of Nodes ###########
'''
When you EVALUATE a node, TensorFlow automatically determines the 
set of nodes that it depends on, and it evaluates those nodes first

Following example: 
TensorFlow automatically detects that y depends on w, 
which depends on x, so it first evaluates w, then x, then y, 
and it returns the value of y. Finally, the code runs the graph to evaluate z. 

Once again, TensorFlow detects that it must first evaluate w and x. 

It is important to note that it will not reuse the result of the 
previous evaluation of w and x. In short, the code below evaluates w and x twice
'''

# Variable
w = tf.constant(3)

# Nodes
x=w+2
y=x+5
z=x*3


with tf.Session() as sess:
    # equivalent to val = sess.run(y)
    print(y.eval()) # 10
    print(z.eval()) # 15

'''
* A variable starts its life when its initializer is run, 
  and it ends when the session is closed.

* Node values are dropped between graph runs
'''


####### Evaluate multiple nodes in the same run ############
'''
If you want to evaluate y and z efficiently, without evaluating 
w and x twice as in the code above, 
you must ask TensorFlow to evaluate both y and z in just one graph run.
'''

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15