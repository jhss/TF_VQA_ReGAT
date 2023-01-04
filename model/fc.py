"""
This code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

import tensorflow as tf
from model.weight_norm import WeightNorm

class FullyConnected(tf.keras.layers.Layer):
  def __init__(self, dims, activation = 'relu', dropout = 0, bias = True):
      super(FullyConnected, self).__init__()

      self.layers = []

      for i in range(len(dims)-2):
          in_dim = dims[i]
          out_dim = dims[i+1]

          if dropout > 0:
             self.layers.append(tf.keras.layers.Dropout(dropout))

          self.layers.append(
            WeightNorm(tf.keras.layers.Dense(out_dim, input_shape=(None, in_dim),
                                             use_bias = bias, activation = None)))

          if activation == 'relu':
              self.layers.append(tf.keras.layers.Activation('relu'))
          elif activation == 'tanh':
              self.layers.append(tf.keras.layers.Activation('tanh'))

      if 0 < dropout:
         self.layers.append(tf.keras.layers.Dropout(dropout))

      self.layers.append(
        WeightNorm(tf.keras.layers.Dense(dims[-1], input_shape=(None, dims[-2]),
                                         use_bias = bias, activation = None)))

      if activation == 'relu':
          self.layers.append(tf.keras.layers.Activation('relu'))
      elif activation == 'tanh':
          self.layers.append(tf.keras.layers.Activation('tanh'))

  def call(self, x):

      for i in range(len(self.layers)):
          x = self.layers[i](x)

      return x
