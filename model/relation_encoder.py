"""
This code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

import tensorflow as tf

from model.fc import FullyConnected
from model.graph_att_net import GraphAttentionNetwork

def concat_visual_question(q, v, mask = True):

    #print("[DEBUG] before q.shape: ", q.shape)
    q = tf.reshape(q, (q.shape[0], 1, q.shape[1]))
    repeat_vals = (q.shape[0], v.shape[1], q.shape[2])
    q = tf.broadcast_to(q, shape = repeat_vals) # [Check]
    #print("[DEBUG] after expand shape: ", q.shape)
    if mask:
        v_sum = tf.reduce_sum(v, axis = -1)
        #print("[DEBUG] v.dtype: ", v.dtype)
        mask_matrix = tf.not_equal(v_sum, tf.constant(0, dtype=tf.float32))
        mask_idx = tf.where(mask_matrix)

        if mask_idx.ndim > 1:
            mask_matrix = tf.cast(mask_matrix, dtype = tf.float32)
            mask_matrix = tf.expand_dims(mask_matrix, axis = -1)
            mask_matrix = tf.broadcast_to(mask_matrix, shape = (mask_matrix.shape[0],
                                                                mask_matrix.shape[1],
                                                                q.shape[2]))
            #print("[DEBUG] q.shape: ", q.shape, " mask_matrix.shape: ", mask_matrix.shape)
            masked_q = q * mask_matrix # [Check]
            #print("[DEBUG] maksed_q: ", masked_q)
            #q[mask_index[:, 0], mask_index[:, 1]] = 0

    # v: [9, 30]
    # maksed_q : [9,30,1024]
    v_cat_q = tf.concat((v, masked_q), axis = -1)

    return v_cat_q

class ImplicitRelationEncoder(tf.keras.layers.Layer):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, pos_emb_dim,
                 nongt_dim, num_heads = 16, num_steps = 1,
                 residual_connection = True, label_bias = True):
        super(ImplicitRelationEncoder, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        print("[DEBUG] implicit out_dim: ", out_dim)
        self.residual_connection = residual_connection
        self.num_steps = num_steps
        print(f"In ImplicitRelationEncoder, num of graph propogate steps: {self.num_steps} residual_connection: {self.residual_connection}")

        if self.v_dim != self.out_dim:
            print("[DEBUG] v2out v_dim, out_dim: ", v_dim, out_dim)
            self.v2out = FullyConnected([v_dim, out_dim])
        else:
            self.v2out = None

        in_dim = out_dim + q_dim
        self.implicit_relation = GraphAttentionNetwork(dir_num, 1, in_dim, out_dim,
                                                       nongt_dim = nongt_dim,
                                                       label_bias = label_bias,
                                                       num_heads = num_heads,
                                                       pos_emb_dim = pos_emb_dim)

    def call(self, visual, pos_emb, question):
        '''
            visual:   [batch, num_rois, v_dim]
            question: [batch, q_dim]
            pos_emb:  [batch, num_rois, nongt_dim, emb_dim] (정확히 뭔지 헷갈림)

            output:   [batch_size, num_rois, out_dim, 3]
        '''

        #print("[DEBUG] implicit input visual.shape: ", visual.shape) # (9, 30, 2048)
        # fully connected due to implicit relation
        batch, num_rois = visual.shape[0], visual.shape[1]
        adj_mat =  tf.ones(shape = (batch, num_rois, num_rois, 1))
        # (9, 30, 2048, 1)
        #print("[DEBUG] visual adj_mat.shape: ", adj_mat.shape)

        if self.v2out:
            visual = self.v2out(visual)
            #print("[DEBUG] after v2out: ", visual.shape)
            # (9, 30, 1024)

        for i in range(self.num_steps):
            v_cat_q = concat_visual_question(question, visual, mask = True)
            imp_rel = self.implicit_relation(v_cat_q, adj_mat, pos_emb)

            if self.residual_connection:
                visual += imp_rel
            else:
                visual = imp_rel

        return visual

class ExplicitRelationEncoder(tf.keras.layers.Layer):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, label_num,
                 nongt_dim = 20, num_heads = 16, num_steps = 1,
                 residiual_connection = True, label_bias = True):
        super(ExplicitRelationEncoder, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        self.residual_connection = residual_connection
        self.num_steps = num_steps
        print(f"In ExplicitRelationEncoder, num of graph propogate steps: {self.num_steps} residual_connection: {self.residual_connection}")

        if self.v_dim != self.out_dim:
            self.v2out = FullyConnected([v_dim, out_dim])
        else:
            self.v2out = None

        in_dim = out_dim + q_dim
        self.explicit_relation = GraphAttentionNetwork(dir_num, label_num, in_dim, out_dim,
                                                       nongt_dim = nongt_dim,
                                                       label_bias = label_bias,
                                                       num_heads = num_heads,
                                                       pos_emb_dim = -1)

    def call(self, visual, adj_mat, question):
        """
        Inputs:
            visual: [batch, num_rois, v_dim]
            question: [batch, q_dim]
            adj_mat: [batch, num_rois, num_rois, num_labels]

        Returns:
            output: [batch, num_rois, out_dim]
        """

        if self.v2out:
            visual = self.v2out(visual)

        for i in range(self.num_steps):
            v_cat_q = concat_visual_question(question, visual, mask = True)
            exp_rel = self.explicit_relation(v_cat_q, adj_mat)

            if self.residual_connection:
                visual += exp_rel
            else:
                visual = exp_rel

        return visual
