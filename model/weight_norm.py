"""
This code is modified by Juhong Song from official TensorFlow repository.
https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/layers/weight_norm.py
MIT License
"""

import tensorflow as tf

class WeightNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):

        if not (type(layer).__name__ == 'Dense' or type(layer).__name__ == 'Conv2D'):
            raise ValueError('WeightNorm is only implemented with Dense and Conv2D layers.')

        super(WeightNorm, self).__init__(layer, **kwargs)

    def build(self, input_shape):

        self.layer.build(input_shape)
        
        self.v = self.add_weight(
                    name = 'v', shape = self.layer.kernel.shape,
                    dtype = self.layer.kernel.dtype, trainable = True)
        
        self.v.assign(self.layer.kernel)
        
        self.g = self.add_weight(
                    name = 'g', shape = [],
                    initializer = 'ones', dtype = self.v.dtype, trainable = True)
        
        self.layer.kernel = None
        self._init_norm()
        super(WeightNorm, self).build()

    def _init_norm(self):
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.v)))
        self.g.assign(kernel_norm)

    def _compute_weights(self):

        self.layer.kernel = tf.nn.l2_normalize(self.v, axis = None) * self.g

    @tf.function
    def call(self, inputs):

        self._compute_weights()
        output = self.layer(inputs)

        return output
