from GradientDescentOptimizer import dimension_separator
from BasicUsage import building_graph
from BasicUsage import interactive_session
from BasicUsage import fetch_tensors
from BasicUsage import feeds
from MNIST import MNIST

print("TensorFlow basics")

print ("1-building tensorflow graph \n2-interactive sessions \n3-fetch tensors \n4-feeds "
       "\n5-Dimension separator\n6-MNIST introduction")
operation = raw_input("Enter the operation number: ")

options = {
    1: building_graph,
    2: interactive_session,
    3: fetch_tensors,
    4: feeds,
    5: dimension_separator,
    6: MNIST.mnist_intro
}

options[int(operation)]()