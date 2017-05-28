import tensorflow as tf


'''
A TensorFlow program is typically split into two parts: 
* the first part builds a computation graph (this is called the construction phase), 
* and the second part runs it (this is the execution phase). 

The construction phase typically builds a computation graph representing the ML model 
and the computations required to train it. 

The execution phase generally runs a loop that evaluates a training step repeatedly 
(for example one step per mini-batch) gradually improving the model parameters.
'''

#### 1 - Construct Computation Graph ###############

# ------- Source Ops ------
'''
Constants and variables take no input (they are called source ops)
'''

'''
Before any of the following code is run, TF generates an
"invisible" default graph.
'''

'''
 Variables are automatically added to the "invisible" graph created
 Note: no computation is performed here
 Declare variables, and give them a value The variables have not been initialized
'''
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
z = tf.placeholder(dtype=tf.int32, name="z")

# --------- Ops ---------
'''
Operations (also called ops for short) can take 
any number of inputs and produce any number of outputs.
Below, we're using the multiplication and addition ops
'''
# Define the function as f(x,y)
f= x * x * y + y + 2
# This is equivalent to the following daisy chaining of ops:
# f = tf.add(tf.add(tf.mul(tf.mul(x, x), y), y), tf.constant(2))

# Function which depends on placeholder
fz = x + z


# Function which depends on placeholder and previous op
fcomplex = fz + f

'''
Ops and source ops are added as 'nodes' in a graph. 
Their inputs/outputs are tensors (more general case of vectors).
So what what gets passed around from one node 
to another are tensors - composed of values
'''

######### 2 - Runing the Graph ###########
'''
To evaluate this "inivsible" default graph you need to open a TensorFlow session 
and use it to initialize the variables and evaluate f.  A TensorFlow session 
takes care of placing the operations onto devices such as CPUs and GPUs and running them, 
and it holds all the variable values
'''

### Sequential, low level way
sess = tf.Session()
try:
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)

    # Equivalent evaluative calls
    print('x.eval(session=sess): {0}'.format(x.eval(session=sess)))
    print('sess.run(x): {0}'.format(sess.run(x)))

    # Run operation that depends on place-holder
    result = sess.run(fz, feed_dict={z:5})
    print('fz: {0}'.format(result))

    # ---- Complex Operation ----
    # Run operation that depends on place-holder and other op
    result = sess.run(fcomplex, feed_dict={z:5})
    print('fcomplex: {0}'.format(result))

    # We can feed whatever we want into the feed_dictionary, and sort of
    # override the default value of the constitutive ops. So, for fcomplex
    # instead of evaluating 'f', feed a value in its place as in following:
    result = sess.run(fcomplex, feed_dict={z: 5, f:1})
    print('fcomplex: {0}'.format(result))

    # ----- Assign any value to a variable ------
    sess.run(tf.assign(x, 45))
    print('after assignment: sess.run(x): {0}'.format(sess.run(x)))
except Exception as ex:
    print(str(ex))
finally:
    sess.close()


#### Briefer form I
'''
Note that 'x' and 'y' are variables. They are initialized.
            A variable starts its life when its initializer is run, 
            and it ends when the session is closed

          'f' is a tensor (a graph node). It is evaluated (it depends on other).
          It returns a value after it is run
          Node values are dropped between graph runs
          
          But init & eval both result in sess.run(object)
'''
# the session is automatically closed at the end of the block
with tf.Session() as sess:
    x.initializer.run() # = tf.get_default_session().run(x.initializer) = sess.run(x.initializer)
    y.initializer.run() # = tf.get_default_session().run(y.initializer) = ... as above
    result2 = f.eval()  # = tf.get_default_session().run(f)  = .. as above

print(result2)


######### Briefer form II #############
'''
it does not actually perform the initialization immediately, 
it creates a node in the graph that will initialize all variables when it is run
'''
init = tf.global_variables_initializer() # prepare an init node

with tf.Session() as sess:
    init.run() # actually initialize ALL the variables
              # (GT: calls the tf.get_default_session().run(x.initializer))
    result3 = f.eval()

print(result3)


######### Interactive Session ##########
init = tf.global_variables_initializer() # prepare an init node
'''
The only difference with a regular Session is that when it is 
created it automatically sets itself as the default session,
so you don’t need a 'with' block. But you must close it
'''
sess = tf.InteractiveSession()
init.run()
result4 = f.eval()
sess.close()