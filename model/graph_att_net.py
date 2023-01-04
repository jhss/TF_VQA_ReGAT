"""
This code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

import tensorflow as tf
from model.fc import FullyConnected
from model.graph_att_layer import GraphSelfAttentionLayer

class GraphAttentionNetwork(tf.keras.layers.Layer):
    def __init__(self, dir_num, label_num, in_feat_dim, out_feat_dim,
                 nongt_dim=20, dropout=0.2, label_bias=True,
                 num_heads=16, pos_emb_dim=-1):

        super(GraphAttentionNetwork, self).__init__()
        assert dir_num <= 2, "Got more than two directions in a graph."
        self.dir_num   = dir_num
        self.label_num = label_num
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.self_weights = FullyConnected([in_feat_dim, out_feat_dim], None, dropout)
        self.bias = FullyConnected([label_num, 1], None, 0.2, label_bias)
        self.nongt_dim = nongt_dim
        self.pos_emb_dim = pos_emb_dim
        self.neighbor_net = []

        for i in range(dir_num):
            g_att_layer = GraphSelfAttentionLayer(
                            pos_emb_dim = pos_emb_dim,
                            num_heads = num_heads,
                            hidden_dim = out_feat_dim,
                            nongt_dim=nongt_dim)
            self.neighbor_net.append(g_att_layer)



    def call(self, v_feat, adj_mat, pos_emb = None):

        if self.pos_emb_dim > 0 and pos_emb is None:
            raise ValueError(
                f"position embedding is set to None "
                f"with pos_emb_dim {self.pos_emb_dim}"
            )
        elif self.pos_emb_dim < 0 and pos_emb is not None:
            raise ValueError(
                f"position embedding is NOT None "
                f"with pos_emb_dim < 0"
            )

        batch_size, num_rois, feat_dim = v_feat.shape
        nongt_dim = self.nongt_dim

        adj_mat_list = [adj_mat, tf.transpose(adj_mat, [0, 2, 1, 3])]

        self_feat = self.self_weights(v_feat)


        output = self_feat
        neighbor_emb = [0] * self.dir_num

        for d in range(self.dir_num):
            input_adj_mat = adj_mat_list[d][:, :, :nongt_dim, :]


            # (9, 30, 20, 1)
            condensed_adj_mat = tf.reduce_sum(input_adj_mat, axis = -1)
            # [9, 30, 20, 1]
            v_biases_neighbors = tf.squeeze(self.bias(input_adj_mat), axis = -1)


            neighbor_emb[d] = self.neighbor_net[d](
                self_feat, condensed_adj_mat, pos_emb, v_biases_neighbors
            )

            output = output + neighbor_emb[d]

        output = self.dropout(output)
        output = tf.nn.relu(output)

        return output
