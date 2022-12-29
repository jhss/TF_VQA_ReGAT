import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Input

from config.parser import parse_with_config
from utils import data_generator
from dataset import Dictionary, VQAFeatureDataset, VQAFeatureDataset2, tfidf_from_questions
from dataset2 import VQAFeatureDatasetFit
from model.rel_graph_net import build_relation_graph_attention_net
from train import train, train_debug, train_fit

def parse_args():
    parser = argparse.ArgumentParser()
    '''
    For training logistics
    '''
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_start', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.25)
    parser.add_argument('--lr_decay_step', type=int, default=2)
    parser.add_argument('--lr_decay_based_on_val', action='store_true',
                        help='Learning rate decay when val score descreases')
    parser.add_argument('--grad_accu_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--save_optim', action='store_true',
                        help='save optimizer')
    parser.add_argument('--log_interval', type=int, default=-1,
                        help='Print log for certain steps')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    '''
    loading trained models
    '''
    parser.add_argument('--checkpoint', type=str, default="")

    '''
    For dataset
    '''
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=["vqa", "vqa_cp"])
    parser.add_argument('--data_folder', type=str, default='../VQA_ReGAT/data')
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true',
                        help='use visual genome dataset to train?')
    parser.add_argument('--adaptive', action='store_true',
                        help='adaptive or fixed number of regions')
    '''
    Model
    '''
    parser.add_argument('--relation_type', type=str, default='implicit',
                        choices=["spatial", "semantic", "implicit"])
    parser.add_argument('--fusion', type=str, default='mutan',
                        choices=["ban", "butd", "mutan"])
    parser.add_argument('--tfidf', action='store_true',
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help="op used in tfidf word embedding")
    parser.add_argument('--num_hid', type=int, default=1024)
    '''
    Fusion Hyperparamters
    '''
    parser.add_argument('--ban_gamma', type=int, default=1, help='glimpse')
    parser.add_argument('--mutan_gamma', type=int, default=2, help='glimpse')
    '''
    Hyper-params for relations
    '''
    # hyper-parameters for implicit relation
    parser.add_argument('--imp_pos_emb_dim', type=int, default=64,
                        help='geometric embedding feature dim')

    # hyper-parameters for explicit relation
    parser.add_argument('--spa_label_num', type=int, default=11,
                        help='number of edge labels in spatial relation graph')
    parser.add_argument('--sem_label_num', type=int, default=15,
                        help='number of edge labels in \
                              semantic relation graph')

    # shared hyper-parameters
    parser.add_argument('--dir_num', type=int, default=2,
                        help='number of directions in relation graph')
    parser.add_argument('--relation_dim', type=int, default=1024,
                        help='relation feature dim')
    parser.add_argument('--nongt_dim', type=int, default=20,
                        help='number of objects consider relations per image')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='number of attention heads \
                              for multi-head attention')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='number of graph propagation steps')
    parser.add_argument('--residual_connection', action='store_true',
                        help='Enable residual connection in relation encoder')
    parser.add_argument('--label_bias', action='store_true',
                        help='Enable bias term for relation labels \
                              in relation encoder')
    parser.add_argument('--print_freq',  default = 500)
    parser.add_argument('--debug', default = 'debug')
    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)
    return args

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    args = parse_args()
    #print("args.seed: ", args.seed)
    print(device_lib.list_local_devices())
    print("[DEBUG] HELLO main main main main")
    #print("args.seed: ", args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    fusion_methods = args.fusion
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


    #args.debug = 'fit'

    #print("[DEBUG] dictionary: ", dictionary)
    #print("[DEBUG] dictionary.idx2word: ", dictionary.idx2word)
    print("[DEBUG] args.debug: ", args.debug)
    if args.debug == 'debug':
        args.data_folder = "../VQA_ReGAT/data/"
        dictionary = Dictionary.load_from_file(
                        os.path.join(args.data_folder, 'glove/dictionary.pkl'))
        print("[DEBUG] args.data_folder: ", args.data_folder)

        val_dset = VQAFeatureDataset(
                    'val', dictionary, args.relation_type, adaptive=args.adaptive,
                    pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder,
                    batch_size = args.batch_size)

        args.model_type = 'normal'
        model = build_relation_graph_attention_net(val_dset, args)
        #del model, val_dset

        tfidf = None
        weights = None
        pt_tfidf   = tf.convert_to_tensor(np.load(args.data_folder + "tfidf.npy"))
        print("[DEBUG] pt_tfidf.shape: ", pt_tfidf.shape)
        pt_weights =  np.load(args.data_folder + "weights.npy")

        if args.tfidf:
            tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dictionary)

            print("[DEBUG] type tfidf: ", type(tfidf))
            #dense_tfidf = tf.sparse.to_dense(tfidf)
            #tfidf_equal = tf.experimental.numpy.isclose(dense_tfidf, pt_tfidf, atol = 1e-5, equal_nan = False)
            #print("[DEBUG] tfidf_equal: ", tf.math.reduce_all(tfidf_equal).numpy())
            #weights_equal =  np.isclose(pt_weights, np_weights, atol = 1e-5, equal_nan = False)
            #print("[DEBUG] weights equal: ", np.all(weights_equal).numpy())

        #print("[DEBUG] tf idf check end")
        #sys.exit()
        model.w_emb.init_embedding(os.path.join(args.data_folder, 'glove/glove6b_init_300d.npy'),
                                   tfidf, weights)

        print("[DEBUG] weight init finish")

        train_debug(model, val_dset, None, args)
        #train_fit(model, )
    elif args.debug == 'tf':
        args.data_folder = "../VQA_ReGAT/data/"
        dictionary = Dictionary.load_from_file(
                        os.path.join(args.data_folder, 'glove/dictionary.pkl'))
        print("[DEBUG] args.data_folder: ", args.data_folder)

        val_dset = VQAFeatureDataset(
                    'val', dictionary, args.relation_type, adaptive=args.adaptive,
                    pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder,
                    batch_size = 64)

        train_dset = VQAFeatureDataset(
                    'train', dictionary, args.relation_type, adaptive=args.adaptive,
                    pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder,
                    batch_size = 32)

        args.model_type = 'normal'
        model = build_relation_graph_attention_net(val_dset, args)

        tfidf = None
        weights = None

        if args.tfidf:
            tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dictionary)

        model.w_emb.init_embedding(os.path.join(args.data_folder, 'glove/glove6b_init_300d.npy'),
                                   tfidf, weights)

        train(model, train_dset, val_dset, args)
    elif args.debug == 'fit':
        args.data_folder = "../VQA_ReGAT/data/"
        dictionary = Dictionary.load_from_file(
                        os.path.join(args.data_folder, 'glove/dictionary.pkl'))
        print("[DEBUG] args.data_folder: ", args.data_folder)

        print("[DEBUG] batch size: ", args.batch_size)
        val_dset = VQAFeatureDatasetFit(
                    'val', dictionary, args.relation_type, adaptive=args.adaptive,
                    args = args, pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder,
                    batch_size = args.batch_size)

        print("[DEBUG] ntoken: ", dictionary.ntoken, " v_dim: ", val_dset.v_dim, " num_ans_candidates: ", val_dset.num_ans_candidates)
        print("[DEBUG] fit start")
        #tf.data.Dataset.from_generator
        #for batch in data_generator(val_dset):
        #    print(batch)
        #    break

        args.model_type = 'fit'
        model = build_relation_graph_attention_net(val_dset, args)
        #del model, val_dset

        tfidf = None
        weights = None
        pt_tfidf   = tf.convert_to_tensor(np.load(args.data_folder + "tfidf.npy"))
        print("[DEBUG] pt_tfidf.shape: ", pt_tfidf.shape)
        pt_weights =  np.load(args.data_folder + "weights.npy")

        if args.tfidf:
            tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dictionary)

            print("[DEBUG] type tfidf: ", type(tfidf))
            #dense_tfidf = tf.sparse.to_dense(tfidf)
            #tfidf_equal = tf.experimental.numpy.isclose(dense_tfidf, pt_tfidf, atol = 1e-5, equal_nan = False)
            #print("[DEBUG] tfidf_equal: ", tf.math.reduce_all(tfidf_equal).numpy())
            #weights_equal =  np.isclose(pt_weights, np_weights, atol = 1e-5, equal_nan = False)
            #print("[DEBUG] weights equal: ", np.all(weights_equal).numpy())

        #print("[DEBUG] tf idf check end")
        #sys.exit()
        model.w_emb.init_embedding(os.path.join(args.data_folder, 'glove/glove6b_init_300d.npy'),
                                   tfidf, weights)

        print("[DEBUG] weight init finish")

        train_fit(model, val_dset, None, args)
    elif args.debug == 'torch':
        print("HERE")

        print("[DEBUG] np weights finish")
        #args.data_folder = '../../../MyDrive/Implementation/VQA_ReGAT/data/'
        #dictionary = Dictionary.load_from_file(
        #                os.path.join(args.data_folder, 'glove/dictionary.pkl'))

        np_weights = np.load("../../../MyDrive/Implementation/VQA_ReGAT/pretrained_models/np_butd.npy", allow_pickle = True)

        ntoken, v_dim, num_ans_candidates =  19901, 2048,  3129

        from model.language_model import WordEmbedding, QuestionEmbedding, QuestionSelfAttention
        from model.relation_encoder import ImplicitRelationEncoder, ExplicitRelationEncoder
        from model.classifier import SimpleClassifier
        from model.fusion import BUTD
        from model.rel_graph_net import RelationGraphAttentionNetwork


        w_emb = WordEmbedding(ntoken, 300, .0, args.op)
        q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
        q_att = QuestionSelfAttention(args.num_hid, .2)

        v_relation = ImplicitRelationEncoder(
                        v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads = args.num_heads, num_steps = args.num_steps,
                        residual_connection = args.residual_connection,
                        label_bias = args.label_bias)

        classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                      num_ans_candidates, 0.5)

        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)

        model = RelationGraphAttentionNetwork(None, w_emb, q_emb, q_att, v_relation,
                                             joint_embedding, classifier, 0,
                                             args.fusion, args.relation_type)

        print("numpy emb shape: ", np_weights.item()['module.w_emb.emb.weight'].shape)
        model.build(input_shape = {"visual": (None, 3, 2, 1),
                                   "b": (None, 3, 1),
                                   "question": (None, 14),
                                   "implicit_pos_emb": (None, None, None, 6),
                                   "sem_adj_mat": (None, 4),
                                   "spa_adj_mat": (None, 5),
                                   "labels": (None, 2)
                                   }
                    )
        for n in model.trainable_variables:
            print("HEllo")
            print(n.shape)

        print("------")

        for n in np_weights.item():
            print("n: ", n, " shape: ", np_weights.item()[n].shape)



    #print(f"[DEBUG] total: {val_dset.data_len}, train {val_dset.train_data_len}, val: {val_dset.val_data_len}")
    #sys.exit()
    # batch size 64 기준 batch_score 30 ~ 40정도 나왔음


    #[CLEAR]


    '''
    output_meta_folder = join(args.output, "regat_%s" % args.relation_type)
    utils.create_dir(output_meta_folder)
    args.output = output_meta_folder+"/%s_%s_%s_%d" % (
                fusion_methods, args.relation_type,
                args.dataset, args.seed)
    if exists(args.output) and os.listdir(args.output):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output))
    utils.create_dir(args.output)
    with open(join(args.output, 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    logger = utils.Logger(join(args.output, 'log.txt'))

    train(model, train_loader, eval_loader, args, device)
    '''
