# imports the script to download mnist data set
from tensorflow.examples.tutorials.mnist import input_data


def mnist_intro():
    # loads the mnist data set. downloads the data set if it's not available
    # in the project path
    # downloaded data is split into three parts. 55000 data points for training,
    # 10000 data point for testing and 5000 for validation.
    # a data point consist of an image of a digit and a label describing the digit.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    import tensorflow as tf

    # a placeholder to input any number of mnist images into a 784 dimensional vector.
    # mnist consist of 28x28 pixel images of digits. [None, 784], none stands to
    # indicate dimension can be any length.
    x = tf.placeholder(tf.float32, [None, 784])

    # variable to keep weights and biases of the model. This can be loaded as inputs,
    # but since we are intended to compute these values at the execution better to keep
    # these as variables.
    # Both W and b will be initialized with zeros. W is defined to keep 10 vectors of
    # size 784 to represent digits 0-9
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # defines the model as a softmax function. First we performs the matrix multiplication
    # of input vector/number and corresponding weight. then add the bias and performs
    # softmax. With tensorflow, model can be represented in a single line like below.
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # to train we perform a cross entropy function. in order to do that another placeholder
    # is require to keep correct answer/label.
    y_ = tf.placeholder(tf.float32, [None, 10])

    # calculate logarithm of each element of y first. multiply each element of y_ with logarithm of y.
    # add the elements in the second dimension of y. finally computes the mean over all the elements
    # in a batch.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # minimize coross entropy using gradient descent optimizer.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # evaluates the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # determines the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    return
