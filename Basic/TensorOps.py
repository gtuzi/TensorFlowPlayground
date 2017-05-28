import tensorflow as tf

'''
Tensor Operations: initializations, constants, variables, shapes, reshaping 
'''

######## Tensors: Varying shapes ###################
a = tf.constant(1.2, dtype=tf.float32, name='a')
b = tf.constant(3.4, dtype=tf.float32, name='b')
c = tf.constant([1.5, 44.3, 55.4], dtype=tf.float32, name='c')
d = tf.constant([[1.5, 44.3, 55.4]], dtype=tf.float32, name='c')
e = tf.constant([[1.5], [44.3], [55.4]], dtype=tf.float32, name='c')
f = tf.constant(4.44, dtype=tf.float32, shape=[4,2], name='d')
g = tf.constant([0.3, 6.7], name='g')
h = tf.constant([4.2, 2.1], name='h')

############ Logical Operations ##########
onesTensor = tf.ones_like(g)
negOnesTensor = (-1)*tf.ones_like(g)
# Compare two tensors... note that these tensors have multiple values
# select() compares items-by-items
sel1 = tf.select(tf.greater_equal(g, h), onesTensor, negOnesTensor, name='sel1')
sel2 = tf.select(tf.less_equal(g, 4.), onesTensor, negOnesTensor, name='sel2')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print('a: {0}, shape{1}'.format(a.eval(), a.get_shape().as_list()))
    print('c: {0}, shape{1}'.format(c.eval(), c.get_shape().as_list()))
    print('d: {0}, shape{1}'.format(d.eval(), d.get_shape().as_list()))
    print('e: {0}, shape{1}'.format(e.eval(), e.get_shape().as_list()))
    print('f: {0}, shape{1}'.format(f.eval(), f.get_shape().as_list()))

    print('')
    print('g: {0}'.format(g.eval()))
    print('h: {0}'.format(h.eval()))
    print('sel1: {0}'.format(sel1.eval()))
    print('')
    print('sel2: {0}'.format(sel2.eval()))

