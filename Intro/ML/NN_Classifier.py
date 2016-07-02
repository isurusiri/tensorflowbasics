import tensorflow as tf
import numpy as np

# data set locations
# training: download.tensorflow.org/data/iris_training.csv
# test: download.tensorflow.org/data/iris_test.csv
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# load data sets
# load_csv() method resides in contrib.learn.datasets requires
# two parameters, file path and input type. In this case the
# input type is numpy int
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)

# variable to the feature data and target values
# x_train for training set feature data
# x_test for test set feature data
# y_train for training set target values
# y_test for test set target values
x_train, x_test, y_train, y_test = training_set.data, test_set.data, training_set.target, test_set.target

# Build 3 layer DNN with 10, 20, 10 neuron units respectively
# with three target classes
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

# Fit model
# set the DNN model according to training data set using
# training data, target value and number of iterations to train
classifier.fit(x=x_train, y=y_train, steps=200)

# Evaluate accuracy
# evaluates the model with test feature data and target values
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))

