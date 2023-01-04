import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from config.parser import parse_with_config
from utils import gen_data_generator
from dataset import Dictionary, VQAFeatureDataset, VQAFeatureDataset2, tfidf_from_questions
from model.rel_graph_net import build_relation_graph_attention_net
from train import train, train_debug

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
    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)
    return args

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    #tf.keras.backend.set_floatx('float64')
    args = parse_args()
    print(device_lib.list_local_devices())

    # seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # load dataset
    fusion_methods = args.fusion
    dictionary = Dictionary.load_from_file(
                    os.path.join(args.data_folder, 'glove/dictionary.pkl'))

    val_dset = VQAFeatureDataset(
                'val', dictionary, args.relation_type, adaptive=args.adaptive,
                pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder,
                batch_size = args.batch_size // 4)

    train_dset = VQAFeatureDataset(
                'train', dictionary, args.relation_type, adaptive=args.adaptive,
                pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder,
                batch_size = args.batch_size)
    
    # load model
    model = build_relation_graph_attention_net(train_dset, args)

    tfidf = None
    weights = None

    # load pre-trained Glove weight
    if args.tfidf:
        tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dictionary)

    model.w_emb.init_embedding(os.path.join(args.data_folder, 
                                            'glove/glove6b_init_300d.npy'),
                               tfidf, weights)
    
    # run train
    train(model, train_dset, val_dset, args)

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
