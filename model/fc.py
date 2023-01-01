"""
This code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

import numpy as np
import tensorflow as tf
from model.weight_norm import WeightNorm

class FullyConnected(tf.keras.layers.Layer):
  def __init__(self, dims, activation = 'relu', dropout = 0, bias = True):
      super(FullyConnected, self).__init__()

      #self.layers = [tfp.layers.weight_norm.WeightNorm(tf.keras.layers.Dense(3, input_shape=(None, 3), use_bias = bias, activation = activation))]
      self.layers = []

      #print("[DEBUG] fc init start")
      for i in range(len(dims)-2):
          #print("[DEBUG] not here")
          in_dim = dims[i]
          out_dim = dims[i+1]

          if dropout > 0:
             self.layers.append(tf.keras.layers.Dropout(dropout))

          #print("[DEBUG] fc in_dim, out_dim: ", in_dim, out_dim)
          self.layers.append(
            WeightNorm(tf.keras.layers.Dense(out_dim, input_shape=(None, in_dim),
                                             use_bias = bias, activation = None)))

          if activation == 'relu':
              self.layers.append(tf.keras.layers.Activation('relu'))
          elif activation == 'tanh':
              self.layers.append(tf.keras.layers.Activation('tanh'))


      #print("[DEBUG] fc here")
      if 0 < dropout:
         self.layers.append(tf.keras.layers.Dropout(dropout))


      #print("[DEBUG] fc dims[-2], dims[-1]: ", dims[-2], dims[-1])
      self.layers.append(
        WeightNorm(tf.keras.layers.Dense(dims[-1], input_shape=(None, dims[-2]),
                                         use_bias = bias, activation = None)))

      if activation == 'relu':
          self.layers.append(tf.keras.layers.Activation('relu'))
      elif activation == 'tanh':
          self.layers.append(tf.keras.layers.Activation('tanh'))

  def call(self, x):

      #if compare:
      #print("[DEBUG] self.fc layers: ", self.layers)
      #print("FC input.shape: ", x.shape)
      for i in range(len(self.layers)):
          #print("[DEBUG] layers: ", self.layers[i])
          x = self.layers[i](x)
          #tf_output = x.numpy()
          #if compare:
              #for p in self.layers.trainable_variables:
              #       print(p.name, p)
              #print(f"[DEBUG] {i}th output compare: ", np.allclose(tf_output, self.pt_output[i], rtol = 1e-5, atol = 1e-5, equal_nan = False))
          #print("[DEBUG] FC output.shape: ", x.shape,)

      return x
