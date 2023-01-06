import tensorflow as tf
import numpy as np
from model.fusion import BUTD
from model.language_model import WordEmbedding, QuestionEmbedding, QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder, ExplicitRelationEncoder
from model.classifier import SimpleClassifier
from model.position_emb import prepare_graph_variables

class RelationGraphAttentionNetwork(tf.keras.Model):
    def __init__(self, w_emb, q_emb, q_att, v_relation,
                 joint_emb, classifier, fusion, relation_type):
        super(RelationGraphAttentionNetwork, self).__init__()
        self.network_name  = f"ReGAT_{relation_type}_{fusion}"
        self.relation_type = relation_type
        self.fusion        = fusion
        self.w_emb         = w_emb
        self.q_emb         = q_emb
        self.q_att         = q_att
        self.v_relation    = v_relation
        self.joint_emb     = joint_emb
        self.classifier    = classifier

    def call(self, visual, bounding_box, question, implicit_pos_emb, 
             sem_adj_mat, spa_adj_mat):
        """
        Inputs:
            visual:            [batch, num_rois, feat_dim]
            bounding_box:      [batch, num_rois, 6]
            question:          [batch, seq_len]
            implicit_pos_emb:  [batch, num_rois, nongt_dim, emb_dim]
            sem_adj_mat:       None if implicit 
                               else [batch, num_rois, num_rois, num_edge_labels]
            spa_adj_mat:       None if implicit
                               else [batch, num_rois, num_rois, num_edge_labels]

        Returns:
            logits
        """

        # Question Embedding: [batch, seq_len] -> [batch, seq_len, w_emb_dim(600)]
        w_emb          = self.w_emb(question)

        # Embed sequence to one feature
        q_emb_seq      = self.q_emb(w_emb)
        q_emb_self_att = self.q_att(q_emb_seq)

        # Apply question-adaptive graph attention to visual feature with relation information.
        if self.relation_type == 'semantic':
            v_emb = self.v_relation(visual, sem_adj_mat, q_emb_self_att)
        elif self.relation_type == 'spatial':
            v_emb = self.v_relation(visual, spa_adj_mat, q_emb_self_att)
        else:
            v_emb = self.v_relation(visual, implicit_pos_emb, q_emb_self_att)

        
        # Embed jointly relation-aware visaul feature and question.
        q_emb = self.q_emb.call_last(w_emb)
        joint_emb, weights = self.joint_emb(v_emb, q_emb)
        
        # Make predictions based on the joint embedding.
        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb

        return logits

def build_relation_graph_attention_net(dataset, args):
    print(f"Building ReGAT model with {args.relation_type} and {args.fusion} fusion method")
    
    dropout = args.dropout

    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, dropout, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, 
                              args.num_hid, 1, False, dropout)
    q_att = QuestionSelfAttention(args.num_hid, dropout)

    if args.relation_type == 'semantic':
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads = args.num_heads,
                        num_steps = args.num_steps, nongt_dim = args.nongt_dim,
                        residual_connection = args.residual_connection,
                        label_bias = args.label_bias)
    elif args.relation_type == 'spatial':
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, arg.relation_dim,
                        args.dir_num, args.spa_label_num,
                        num_heads = args.num_heads,
                        num_steps = args.num_steps, nongt_dim = args.nongt_dim,
                        residual_connection = args.residual_connection,
                        label_bias = args.label_bias)
    else:
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads = args.num_heads, num_steps = args.num_steps,
                        residual_connection = args.residual_connection,
                        label_bias = args.label_bias)

    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, dropout)


    joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)

    model =  RelationGraphAttentionNetwork(w_emb, q_emb, q_att, v_relation,
                                          joint_embedding, classifier,
                                          args.fusion, args.relation_type)

    # one forward pass to build model
    if args.mode == 'eval':
        inputs = dataset.split_entries(0)
        visual_feature, norm_bb, question, bb, spa_adj_matrix, sem_adj_matrix, _ = inputs
        num_objects = visual_feature.shape[1]

        pos_emb, sem_adj_mat, spa_adj_mat = prepare_graph_variables(
                    dataset.relation_type, bb, sem_adj_matrix, spa_adj_matrix, 
                    num_objects, args.nongt_dim, args.imp_pos_emb_dim, 
                    args.spa_label_num, args.sem_label_num)

        model(visual_feature, norm_bb, question, pos_emb, sem_adj_mat, spa_adj_mat)

    return model
