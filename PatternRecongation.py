# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 01:01:37 2017

@author: Daniel Lukic
"""

from __future__ import print_function
import tensorflow as tf
import tensorfunctions as tf_function
import matplotlib.pyplot as plt
import numpy as np

data_train_in = np.transpose([[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
data_train_out = np.array([[0], [1], [1], [0]])

data_test_in = np.transpose([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
data_test_out = np.array([[1], [1], [0], [0]])

learning_rates = [0.01, 0.001]
training_epochs = 100000
display_step = 100
n_input = 3
n_classes = 1

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Create model
pred = tf_function.multilayer_perceptron(x, nn_info=[n_input, 4, n_classes])

# List for Convergence Epochs
conv_epoch_all = []
# List for Training Errors
error_train = []
# List for Testing Errors
error_test = []

# Train for different learning rates
for learning_rate in learning_rates:
    # Define loss and optimizer
    cost = tf.losses.mean_squared_error(y, pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate
                                       ).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    cee_train = []
    cee_test = []

    print('Training for lr = ' + str(learning_rate))
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={x: data_train_in,
                            y: data_train_out})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=",
                      "{:.9f}".format(c))

            err = sess.run(cost, feed_dict={x: data_train_in,
                                            y: data_train_out})
            cee_train.append(err)
            err = sess.run(cost, feed_dict={x: data_test_in,
                                            y: data_test_out})
            cee_test.append(err)

            if epoch > 1:
                if np.abs(cee_test[epoch] - cee_test[epoch - 1]) < 0.0000001:
                    print('Break Test Error Convergence')
                    break

        conv_epoch_all.append(epoch)
        error_train.append(cee_train)
        error_test.append(cee_test)
        print("Optimization Finished!")
        print("Reference")
        print(data_test_out)
        print('')
        print("Prediction")
        print(sess.run(pred, feed_dict={x: data_test_in}))
        print('----------------------------------')
        print('')

# Plotting the Errors
number_of_subplots = 2
plt.figure(figsize=(5, 3), dpi=100)

for v in np.arange(number_of_subplots):
    ax1 = plt.subplot(2, 1, v + 1)
    ax1.semilogx(error_train[v], 'r')
    ax1.semilogx(error_test[v], 'b')

    plt.legend(['MSE Train', 'MSE Test'], fontsize=8)
    plt.xlabel('epoch', fontsize=8)
    plt.ylabel('MSE', fontsize=8)
    plt.title('MSE for lr = ' + str(learning_rates[v]) +
              ', epoch conv. = ' + str(conv_epoch_all[v]), fontsize=8)
    plt.grid()
plt.show()
