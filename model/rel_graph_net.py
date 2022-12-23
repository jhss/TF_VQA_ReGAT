import tensorflow as tf
from model.fusion import BUTD
from model.language_model import WordEmbedding, QuestionEmbedding, QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder, ExplicitRelationEncoder
from model.classifier import SimpleClassifier

class RelationGraphAttentionNetwork(tf.keras.Model):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                 joint_emb, classifier, glimpse, fusion, relation_type):
        super(RelationGraphAttentionNetwork, self).__init__()
        self.network_name  = f"ReGAT_{relation_type}_{fusion}"
        self.relation_type = relation_type
        self.fusion        = fusion
        self.dataset       = dataset
        self.glimpse       = glimpse
        self.w_emb         = w_emb
        self.q_emb         = q_emb
        self.q_att         = q_att
        self.v_relation    = v_relation
        self.joint_emb     = joint_emb
        self.classifier    = classifier

    def call(self, visual, b, question, implicit_pos_emb, sem_adj_mat,
                spa_adj_mat, labels):

        #print("[DEBUG] relation graph vsiual.shape: ", visual.shape) # [9, 30, 2048]
        #print("[DEBUG] question.shape: ", question.shape) # (9, 14)
        w_emb          = self.w_emb(question)
        #print("[DEBUG] w_emb.shape: ", w_emb.shape) # (9, 14, 600)
        q_emb_seq      = self.q_emb(w_emb) # forward_all -> call
        #print("[DEBUG] q_emb_seq.shape: ", q_emb_seq.shape)
        # q_emb_seq.shape:  (9, 1024)
        q_emb_self_att = self.q_att(q_emb_seq)

        if self.relation_type == 'semantic':
            v_emb = self.v_relation(visual, sem_adj_mat, q_emb_self_att)
        elif self.relation_type == 'spatial':
            v_emb = self.v_relation(visual, spa_adj_mat, q_emb_self_att)
        else:
            v_emb = self.v_relation(visual, implicit_pos_emb, q_emb_self_att)

        #print("[DEBUG] joint emb w_emb.shape: ", w_emb.shape) # (9, 14, 600)
        q_emb = self.q_emb.call_last(w_emb)
        #print("[DEBUG] joint emb q_emb: ", q_emb.shape) # (9, 1024)
        joint_emb, weights = self.joint_emb(v_emb, q_emb)

        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb

        return logits, weights

def build_relation_graph_attention_net(dataset, args):
    print(f"Building ReGAT model with {args.relation_type} and {args.fusion} fusion method")

    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
    q_att = QuestionSelfAttention(args.num_hid, .2)

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
        #print("[DEBUG] args.relation_dim: ", args.relation_dim)
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads = args.num_heads, num_steps = args.num_steps,
                        residual_connection = args.residual_connection,
                        label_bias = args.label_bias)

    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)

    gamma = 0

    joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)

    return RelationGraphAttentionNetwork(dataset, w_emb, q_emb, q_att, v_relation,
                                         joint_embedding, classifier, gamma,
                                         args.fusion, args.relation_type)
