import tensorflow as tf
import tensorflow_probability as tfp

class SimpleClassifier(tf.keras.layers.Layer):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.layers = [
            #tfp.layers.weight_norm.WeightNorm(tf.keras.layers.Dense(hid_dim, input_shape = (in_dim, ))),
            tf.keras.layers.Dense(hid_dim, input_shape = (in_dim, )),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout),
            #tfp.layers.weight_norm.WeightNorm(tf.keras.layers.Dense(out_dim)),
            tf.keras.layers.Dense(out_dim)
        ]

    def call(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
