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

## Note: no computation is performed here
# Declare variables, and give them a value
# Note: the variables have not been initialized
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

# Define the function as f(x,y)
f=x*x*y+y+2

######### 2 - Runing the Graph ###########
'''
To evaluate this graph you need to open a TensorFlow session 
and use it to initialize the variables and evaluate f. 
A TensorFlow session takes care of placing the operations 
onto devices such as CPUs and GPUs and running them, 
and it holds all the variable values
'''

### Sequential, low level way
sess = tf.Session()
try:
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)
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
init = tf.global_variable_initializers() # prepare an init node

with tf.Session() as sess:
    init.run() # actually initialize ALL the variables
              # (GT: calls the tf.get_default_session().run(x.initializer))
    result3 = f.eval()

print(result3)


######### Interactive Session ##########
init = tf.global_variable_initializers() # prepare an init node
'''
The only difference with a regular Session is that when it is 
created it automatically sets itself as the default session,
so you don’t need a 'with' block. But you must close it
'''
sess = tf.InteractiveSession()
init.run()
result4 = f.eval()
sess.close()