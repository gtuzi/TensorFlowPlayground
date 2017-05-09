import tensorflow as tf


# Any node you create is automatically added to the DEFAULT graph
x1 = tf.Variable(1)

print('x1.graph is default: {0} '.format(x1.graph is tf.get_default_graph()))



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
