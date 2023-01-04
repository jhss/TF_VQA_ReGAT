"""
This code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

import tensorflow as tf

from model.fc import FullyConnected

class BUTD(tf.keras.layers.Layer):
    def __init__(self, v_dim, q_dim, hidden_dim, dropout = 0.2):
        super(BUTD, self).__init__()
        self.v2attention    = FullyConnected([v_dim, hidden_dim], dropout)
        self.q2attention    = FullyConnected([q_dim, hidden_dim], dropout)
        self.dropout        = tf.keras.layers.Dropout(dropout)
        self.linear         = FullyConnected([hidden_dim, 1], dropout)
        self.visual_embed   = FullyConnected([v_dim, hidden_dim], dropout)
        self.question_embed = FullyConnected([q_dim, hidden_dim], dropout)

    def call(self, visual, question):
        """
        Inputs:
            visual:   [batch, num_rois, v_dim]
            question: [batch, q_dim]

        Outputs:
            joint_emb: [batch, v_dim]
        """

        weights = self.attention_weights(visual, question)
        # [batch, num_rois, v_dim] -> [batch, v_dim]
        weighted_visual = tf.reduce_sum(weights * visual, axis = 1)

        weighted_visual = self.visual_embed(weighted_visual)
        question        = self.question_embed(question)

        joint_emb = weighted_visual * question

        return joint_emb, weights

    def attention_weights(self, visual, question):

        _, k, _ = visual.shape

        visual   = self.v2attention(visual)
        question = self.q2attention(question)
        question = tf.tile(question[:, tf.newaxis, :], (1, k, 1))

        joint = self.dropout(visual * question)
        joint = self.linear(joint)

        return tf.nn.softmax(joint, axis = 1)
