# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 01:01:16 2017

@author: Daniel Lukic
"""

import tensorflow as tf
import numpy as np


def weight_variable(shape):
    """This function constructs a tensorflow variable for a given shape
        with nummerical values which are dstributed truncated normal

    Args:
        shape: Shape of TensorFlow variable.

    Returns:
        Returns weigts for as a TensorFlow variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """This function constructs a tensorflow variable for a given shape
        with nummerical values which are dstributed truncated normal

    Args:
        shape: Shape of TensorFlow variable.

    Returns:
        Returns bias for as a TensorFlow variable
    """

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def multilayer_perceptron(x, nn_info):
    """This function constructs the the Neural Net.
    It matmuls the layers with the given nodes together and
    adds the weights to the nodes. Also the activation functon is
    added to the hidden layers

    Args:
        x: TensorFlow placeholder of input layer with number of nodes.
        nn_info: Nodes for input layer, hidden layer(s) and output layer.
        Also number of hidden layer.

    Returns:
        net: Returns a constructed neural net tensor

    """
    dimension_Input = nn_info[0]
    dimension_Target = nn_info[np.size(nn_info)-1]

    if np.size(nn_info) == 3:
        layers_info = [nn_info[0]]
    else:
        layers_info = nn_info[1:np.size(nn_info)-2]

    weights = weight_variable([dimension_Input, layers_info[0]])
    bias = bias_variable([layers_info[0]])

    layer = tf.add(tf.matmul(x, weights), bias)
    lenght_layers = np.size(layers_info)

    if lenght_layers > 1:
        for i in np.arange(lenght_layers-1):
            weights = weight_variable([layers_info[i], layers_info[i+1]])
            bias = bias_variable([layers_info[i+1]])
            # Hidden layer with tanh activation
            layer = tf.add(tf.matmul(layer, weights), bias)
            layer = tf.tanh(layer)

    # Output layer with linear activation
    weights_out = weight_variable([layers_info[lenght_layers - 1],
                                   dimension_Target])
    bias_out = bias_variable([dimension_Target])
    net = tf.tanh(tf.matmul(layer, weights_out) + bias_out)

    return net
