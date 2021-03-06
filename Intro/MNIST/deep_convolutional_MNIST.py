import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# downloads mnist data set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start an interactive tensorflow session. usually tensorflow
# calls a connection that it holds to native code as a session
# therefore in order to execute the graph computation it is
# necessary to make the graph in advance and the start it in a
# tensorflow session. with interactive session it is possible to
# add computations to graph on the fly and make necessary
# modifications
sess = tf.InteractiveSession()

# placeholders to keep the values of input images and target output
# classes. x will consist of a 2d tensor of floating point numbers.
# 784 is the dimensionality of a single image and none indicates the
# first dimension. y is also a 2d tensor where each row is a one-hot
# 10 dimensional vector
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# defines weights and bias as variables since those can be modified
# as the computation proceeds
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initializes the variables before executes the graph
sess.run(tf.initialize_all_variables())

# multiply the vectorized input images x by the weight matrix W, add
# the bias b, and compute the softmax probabilities that are assigned to each class
y = tf.nn.softmax(tf.matmul(x,W) + b)

# cost function to be minimized during training can be specified just as easily.
# cost function will be the cross-entropy between the target and the model's prediction
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# use steepest gradient descent, with a step length of 0.5, to descend the cross entropy
# add new operations to the computation graph. These operations included ones to compute
# gradients, compute parameter update steps, and apply update steps to the parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# apply the gradient descent updates to the parameters
# trains the model with train_step
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# finds the correctly predicted label. tf.argmax gives the index of the highest
# entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))