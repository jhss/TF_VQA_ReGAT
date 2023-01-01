import tensorflow as tf
from model.weight_norm import WeightNorm

class SimpleClassifier(tf.keras.layers.Layer):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.layers = [
            WeightNorm(tf.keras.layers.Dense(hid_dim, input_shape = (in_dim, ))),
            #tf.keras.layers.Dense(hid_dim, input_shape = (in_dim, )),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout),
            WeightNorm(tf.keras.layers.Dense(out_dim)),
            #tf.keras.layers.Dense(out_dim)
        ]

    def call(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
