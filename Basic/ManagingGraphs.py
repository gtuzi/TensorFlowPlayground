import tensorflow as tf


# Any node you create is automatically ADDED to the DEFAULT graph
x1 = tf.Variable(1)
print('x1.graph is default: {0} '.format(x1.graph is tf.get_default_graph())) # Prints True


#### Manage multiple independent graphs ####
'''
Create a new Graph and temporarily making it 
the default graph inside a with block
'''
sgraph = tf.Graph()
with sgraph.as_default():
    # 'sgraph' is now the default here
    x2 = tf.Variable(2)

print('x2.graph is sgraph: {0}'.format(x2.graph is sgraph)) # Prints True
print('x1.graph is x2.graph: {0}'.format(x1.graph is x2.graph)) # Prints False
print('x1.graph is tf.get_default_graph(): {0}'.format(x1.graph is tf.get_default_graph()))
print('-------------------------')

##### Graphs and sessions ##########

# 1 - Run two separate graphs in separate sessions
'''
If you are using more than one graph (created with `tf.Graph()` in the same
process, you will have to use different sessions for each graph.
'''
tf.reset_default_graph()

graph1 = tf.Graph()
with graph1.as_default():
    # Assign v1 to graph1
    v1 = tf.Variable(3.33, dtype=tf.float32, name='v1')

graph2 = tf.Graph()
with graph2.as_default():
    # Assign v2 to graph2
    v2 = tf.Variable(666.2, dtype=tf.float32, name='v2')

print('v1.graph is v2.graph: {0}'.format(v1.graph is v2.graph)) # False
# We are out of the scope where graph2 was the default. So this will not be true
print('v2.graph is the default graph: {0}'.format(v2.graph is tf.get_default_graph())) # False

# We set graph2 as the default graph here
with graph2.as_default():
    # By passing graph=None, we are saying that this
    # session will run the default graph
    sess = tf.Session(graph=None)
    try:
        sess.run(v2.initializer)
        print('v2: {0:.2f}'.format(sess.run(v2)))

        # This throws an exception because
        # v1 is a node in graph1 which is not the
        # default graph now
        # sess.run(v1.initializer) : Exception !!!
        # print('v1: {0}'.format(sess.run(v1)))
    except Exception as ex:
        print(str(ex))
    finally:
        sess.close()

print('-------------------------')

# 2 - Run the same graph in multiple sessions
'''
Separate graph can be used in multiple sessions. In this case, it
is often clearer to pass the graph to be launched explicitly to
the session constructor.
'''

tf.reset_default_graph()

with graph1.as_default():
    # Add the following ops to graph1
    v3 = tf.Variable(12.9, dtype=tf.float32)
    f1 = tf.assign(v3, v3 + 4)
    f2 = tf.assign(v3, v3 + 40)

    # Create two sessions
    sess1 = tf.Session(graph=None)
    sess2 = tf.Session(graph=None)

    try:
        # sess1 runs graph1
        sess1.run(v3.initializer)
        print('sess1 --> v3: {0:.3f}'.format(sess1.run(v3)))
        sess1.run(f1)
        print('sess1 --> v3: {0:.3f}'.format(sess1.run(v3)))

        # sess2 runs graph1
        sess2.run(v3.initializer)
        print('sess2 --> v3: {0:.3f}'.format(sess2.run(v3)))
        sess2.run(f2)
        print('sess2 --> v3: {0:.3f}'.format(sess2.run(v3)))
    except Exception as ex:
        print(str(ex))
    finally:
        sess1.close()
        sess2.close()

