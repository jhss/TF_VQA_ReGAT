"""
This code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from model.fc import FullyConnected

class GraphSelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nongt_dim, pos_emb_dim = -1,
                 num_heads = 16, dropout = [0.2, 0.5]):
      super(GraphSelfAttentionLayer, self).__init__()

      self.num_heads = num_heads
      self.hidden_dim = hidden_dim
      self.pos_emb_dim = pos_emb_dim

      self.fc_dim   = num_heads
      self.head_dim = int(hidden_dim / num_heads)
      if self.pos_emb_dim > 0:
          self.pair_pos_fc = FullyConnected([pos_emb_dim, self.fc_dim], activation = None, dropout = dropout[0])

      self.query = FullyConnected([hidden_dim, hidden_dim], None, dropout[0])
      self.key   = FullyConnected([hidden_dim, hidden_dim], None, dropout[0])
      self.nongt_dim = nongt_dim

      self.linear_out_ = tf.keras.layers.Conv2D(
                                filters = hidden_dim, # out_channels
                                input_shape = (self.fc_dim * hidden_dim,),
                                kernel_size = (1, 1),
                                groups = self.fc_dim)

    def call(self, roi, adj_mat, pos_emb, label_att):
        batch_size = roi.shape[0]
        num_rois   = roi.shape[1]
        nongt_dim  = self.nongt_dim if self.nongt_dim < num_rois else num_rois
        trunc_rois = roi[:, :nongt_dim, :]

        # [b, num_rois, hidden_dim]
        query = self.query(roi)

        # [b, num_rois, num_heads, hidden_dim / num_heads]
        split_query = tf.reshape(query, (batch_size, num_rois, self.num_heads, self.head_dim))
        # [b, num_heads, num_rois, hidden_dim / num_heads]
        split_query = tf.transpose(split_query, [0, 2, 1, 3])

        key       = self.key(trunc_rois)
        split_key = tf.reshape(key, (batch_size, nongt_dim, self.num_heads, self.head_dim))
        split_key = tf.transpose(split_key, [0, 2, 1, 3])

        # [batch, nongt_dim, hidden_dim]
        value     = trunc_rois

        aff = (1.0 / tf.math.sqrt(float(self.head_dim))) * tf.matmul(split_query, tf.transpose(split_key, [0, 1, 3, 2]))
        # [batch, num_rois, num_heads, nongt_dim]
        weighted_aff = tf.transpose(aff, [0, 2, 1, 3])

        # implicit relation
        if pos_emb is not None and self.pos_emb_dim > 0:
            #print("[DEBUG] pos_emb is not None: ", )
            #pos_emb = pos_emb.float()
            #print("[DEBUG] pos_emb dtype : ", pos_emb.dtype)
            # [batch_size, num_rois * nongt_dim, emb_dim]
            pos_emb = tf.reshape(pos_emb, (batch_size, -1, self.pos_emb_dim))

            # [batch, num_rois * nongt_dim, fc_dim] (emb_dim -> fc_dim)
            pos_emb    = tf.nn.relu(self.pair_pos_fc(pos_emb))
            # [batch, num_rois, nongt_dim, fc_dim] ( num_heads = fc_dim)
            pos_weight = tf.reshape(pos_emb, (batch_size, -1, nongt_dim, self.num_heads))
            # [batch, num_rois, fc_dim, nongt_dim]
            pos_weight = tf.transpose(pos_weight, [0, 1, 3, 2])

            threshold  = tf.constant([1e-6])
            # print("[DEBUG] threshold device type: ", threshold.device)
            pos_weight = tf.math.maximum(pos_weight, threshold)

            weighted_aff += tf.math.log(pos_weight)

        if adj_mat is not None:
            #print("[DEBUG] adj mat is not None")
            aff_transpose = tf.transpose(weighted_aff, [0, 1, 3, 2])
            mask = -9e15 * tf.ones_like(aff_transpose)

            adj_mat = tf.expand_dims(adj_mat, axis = -1)
            #print("[DEBUG] Before adj_mat.shape: ", adj_mat.shape)
            adj_mat = tf.broadcast_to(adj_mat, [adj_mat.shape[0], adj_mat.shape[1],
                                                adj_mat.shape[2], aff_transpose.shape[-1]])

            #print("[DEBUG] adj_mat.shape: ", adj_mat.shape)
            masked_aff = tf.where(adj_mat > 0, aff_transpose, mask)

            masked_aff = masked_aff + tf.expand_dims(label_att, axis = 3)
            weighted_aff = tf.transpose(masked_aff, [0, 1, 3, 2])

        final_aff = tf.nn.softmax(weighted_aff, axis = 3)
        # [batch_size, num_rois * fc_dim, nongt_dim]
        final_aff = tf.reshape(final_aff, (batch_size, -1, nongt_dim))

        # [batch, num_roi * fc_dim, hidden_dim]
        self_att  = tf.matmul(final_aff, value)
        self_att  = tf.reshape(self_att, (batch_size, num_rois, self.fc_dim, self.hidden_dim))
        self_att  = tf.reshape(self_att, (batch_size * num_rois, self.fc_dim * self.hidden_dim))
        self_att  = self_att[:, tf.newaxis, tf.newaxis, :]

        # (270, 1, 1, 16384)
        # tensorflow conv2 input shape [batch, height, width, channel]
        #print("[DEBUG] self_att.shape: ", self_att.shape)
        output = self.linear_out_(self_att)
        # (270, 1, 1, 1024)
        #print("[DEBUG] linear_output.shape: ", output.shape)
        reshape_output = tf.reshape(output, (batch_size, num_rois, self.hidden_dim))
        # (9, 30, 1024)
        #print("[DEBUG] reshape output: ", reshape_output.shape)
        return tf.reshape(output, (batch_size, num_rois, self.hidden_dim))
