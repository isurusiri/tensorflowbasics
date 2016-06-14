import tensorflow as tf


# create a constant op that produces a 1x2 matrix. The op
# is added as a node to the default graph.
#
# the value returned by the constructor represents the output
# of the constant op.
def building_graph():
    matrix_1 = tf.constant([[3., 3.]])

    # create another constant op that produces a 2x1 matrix.
    matrix_2 = tf.constant([[2.], [2.]])

    # create a mutual op that takes matrix_1 and matrix_2 as inputs.
    # the returned value, 'product' represents the result of the
    # matrix multiplication.
    product = tf.matmul(matrix_1, matrix_2)

    # Launch the default graph
    sess = tf.Session()

    # to run the matmul op call the run method of the session, passing
    # 'product' which represents the output of the matmul op. this
    # indicates to the call that we want to get the output of the matmul
    # op back.
    #
    # all inputs needed by the op are run automatically by the session.
    # they typically are run in parallel.
    #
    # the call run(product) causes the execution of three ops  in the
    # graph: the two constant ops and matmul op.
    # the output of the op is returned in 'result' as a numpy 'ndarray' object.
    result = sess.run(product)
    print(result)
    # obvious answer should be [[12.]]

    # close the session. this will release all resources.
    sess.close()
    return


# enter intractive tensorflow session
def interactive_session():
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])

    # initialize 'x' using the run() method of its initializer op.
    x.initializer.run()

    # add an op to subtract 'a' from 'x'. run it and print the result.
    sub = tf.sub(x, a)
    print(sub.eval())
    # obvious answer should be [-2, -1]

    # close the session. this will release all resources.
    sess.close()

    # create a variable that will be initialize to scalar value 0.
    state = tf.Variable(0, name="counter")

    # create an op that add one to state.
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # variables must be initialized by running an 'init' op after having
    # launch the graph. first add the init op to the graph.
    init_op = tf.initialize_all_variables()

    # launch the graph and run the ops.
    with tf.Session() as sess:
        # run the init() op
        sess.run(init_op)
        # print the initial value of the state
        print(sess.run(state))
        # run the op that updates the state and print that value
        for _ in range(5):
            sess.run(update)
            print(sess.run(state))

    # obvious output should be:
    # 0
    # 1
    # 2
    # 3
    # 4
    # 5
    return


# fetch state of multiple tensors after performing op by
# session run() function
def fetch_tensors():
    input_1 = tf.constant([3.0])
    input_2 = tf.constant([2.0])
    input_3 = tf.constant([5.0])
    inter_med = tf.add(input_2, input_3)
    mul = tf.mul(input_1, inter_med)

    with tf.Session() as sess:
        result = sess.run([mul, inter_med])
        print(result)

    # output should be something like [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
    return


# assign placeholders to variables and passes values to them as a feed in session run
def feeds():
    input_1 = tf.placeholder(tf.float32)
    input_2 = tf.placeholder(tf.float32)
    output = tf.mul(input_1, input_2)

    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input_1:[7.], input_2:[2.]}))

    # output should be [array([ 14.], dtype=float32)]
    return


print ("1-building tensorflow graph \n2-interactive sessions \n3-fetch tensors\n4-feeds")
operation = raw_input("Enter the operation number: ")

options = {
    1: building_graph,
    2: interactive_session,
    3: fetch_tensors,
    4: feeds
}

options[int(operation)]()



