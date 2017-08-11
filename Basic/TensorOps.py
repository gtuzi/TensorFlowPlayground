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
i = tf.constant([[[1, 11, 111], [2, 22, 222]],
                 [[3, 33, 333], [4, 44, 444]],
                 [[5, 55, 555], [6, 66, 666]]],
                dtype=tf.int32, name='i')

############ Logical Operations ##########
onesTensor = tf.ones_like(g)
negOnesTensor = (-1)*tf.ones_like(g)
# Compare two tensors... note that these tensors have multiple values
# select() compares items-by-items
sel1 = tf.select(tf.greater_equal(g, h), onesTensor, negOnesTensor, name='sel1')
sel2 = tf.select(tf.less_equal(g, 4.), onesTensor, negOnesTensor, name='sel2')

init = tf.global_variables_initializer()


############ Slicing and Dicing ############
slice_op = tf.slice(i, begin=[1,1,0], size=[2, 1, 2])
# Reverse slicing requires negative stride
strided_slice_op = tf.strided_slice(i, begin=[-1,0,0], end=[-3, 2, 3], strides=[-1, 1, 2])
with tf.Session() as sess:
    init.run()

    print('----------- Shape and Values --------------')
    print('a: {0}\n shape{1}\n\n'.format(a.eval(), a.get_shape().as_list()))
    print('c: {0}\n shape{1}\n\n'.format(c.eval(), c.get_shape().as_list()))
    print('d: {0}\n shape{1}\n\n'.format(d.eval(), d.get_shape().as_list()))
    print('e: {0}\n shape{1}\n\n'.format(e.eval(), e.get_shape().as_list()))
    print('f: {0}\n shape{1}\n\n'.format(f.eval(), f.get_shape().as_list()))
    print('i: {0}\n shape{1}\n\n'.format(i.eval(), i.get_shape().as_list()))

    print('---------- Selection --------------')
    print('g: {0}\n'.format(g.eval()))
    print('h: {0}\n'.format(h.eval()))
    print('sel1: {0}\n'.format(sel1.eval()))
    print('sel2: {0}\n'.format(sel2.eval()))

    print('---------- Slicing ----------------')
    slice_val = slice_op.eval()
    strided_slice_val = strided_slice_op.eval()
    print('Slice_op:\n{0}\n-------\nshape{1}\n\n'.format(slice_val, slice_val.shape))
    print('Strided Slice_op:\n{0}\n-------\nshape{1}\n\n'.format(strided_slice_val, strided_slice_val.shape))

