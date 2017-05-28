import tensorflow as tf


######### Lifecycle of Nodes ###########
'''
In a graph:
When you EVALUATE a node, TensorFlow automatically determines the 
set of nodes that it depends on, and it evaluates those nodes first

Following example: 
TensorFlow automatically detects that y depends on x, 
which depends on w, so it first evaluates w, then x, then y, 
and it returns the value of y. Finally, the code runs the graph to evaluate z. 
Once again, TensorFlow detects that it must first evaluate w and x. 
It is important to note that it will not reuse the result of the 
previous evaluation of w and x. In short, the code below evaluates w and x twice
'''


'''
- Constants and variables take no input (they are called source ops)

- A variable starts its life when its initializer is run, 
  and it ends when the session is closed.
'''
# Variable (source op)
# This is automatically added to the default graph (invisible)
w = tf.constant(3)


'''
- Operations (also called ops for short) can take 
  any number of inputs and produce any number of outputs
- Operations can be source operations (source ops) or just ops.
- Ops are addition (+ or tf.add()), subtraction (- or tf.sub()), etc
- Node values are dropped between graph runs
'''
# Nodes
x=w+2
y=x+5
z=x*3

# When the graph is not specified for a session, it uses
# the default graph to run
with tf.Session() as sess:
    # equivalent to val = sess.run(y)
    print(y.eval()) # 10
    print(z.eval()) # 15

    # Are these nodes part of the same graph ?
    print('y.graph is z.graph: {0}'.format(y.graph is z.graph)) # True

    # Are these nodes part of the default graph ?
    print('y.graph is the default graph: {0}'.format(y.graph is tf.get_default_graph()))  # True



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

