"""
This code is modified by Juhong Song from official TensorFlow repository.
https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/layers/weight_norm.py
MIT License
"""

import tensorflow as tf

class WeightNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        #self.kernel = layer.kernel
        if type(layer).__name__ == 'Dense':
            self.kernel_len = 2
        elif type(layer).__name__ == 'Conv2D':
            self.kernel_len = 4
        else:
            raise ValueError('WeightNorm is only implemented with Dense and Conv2D layers.')
        #self.kernel_len = len(self.kernel.shape)
        #print("[DEBUG] self.kernel.shape: ", self.kernel.shape)
        self.axis = [i for i in range(self.kernel_len)]
        #print("[DEBUG] weight norm init kernel: ", K.eval(self.kernel))
        super(WeightNorm, self).__init__(layer, **kwargs)

    def build(self, input_shape):

        #self.layer.build(input_shape)
        #print("[DEBUG] after build: ", K.eval(self.layer.kernel))
        self.v = self.layer.kernel
        self.layer.kernel = None
        self.g = self.add_weight(
                    name = 'g', shape = [],
                    initializer = 'ones', dtype = self.v.dtype, trainable = True)

        super(WeightNorm, self).build()

    def _init_norm(self):
        #print("[DEBUG] self.layer.kernel: ", K.eval(self.v), " kernel shape: ", self.v.shape)
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.v)))
        #print("[DEBUG] kernel_norm: ", K.eval(kernel_norm))
        self.g.assign(kernel_norm)

    def _compute_weights(self):

        self.layer.kernel = tf.nn.l2_normalize(self.v, axis = self.axis) * self.g
        #print("[DEBUG] self.g: ", self.g)
        #print("[DEBUG] l2_norm: ", tf.math.l2_normalize(self.v, axis = self.axis))


    @tf.function
    def call(self, inputs):
        self._init_norm()
        self._compute_weights()
        output = self.layer(inputs)

        return output
