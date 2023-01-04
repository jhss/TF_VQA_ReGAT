"""
This code is modified by Juhong from Linjie Li's repository.
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

from __future__ import print_function
import os
import json
import pickle
import h5py
import itertools
import math
import utils

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '')\
            .replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK
                # for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries
    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
        (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))
    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries

def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['image_id'] == vgv_id:
            return v['coco_id']
    return None

class VQAFeatureDataset:
    def __init__(self, name, dictionary, relation_type, batch_size, dataroot='data',
                 adaptive=False, pos_emb_dim=64, nongt_dim=36):
        super(VQAFeatureDataset, self).__init__()

        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = adaptive
        self.batch_size = batch_size

        self.aligned_features = []
        self.aligned_normalized_bbs = []
        self.questions = []
        self.targets   = []
        self.question_ids = []                 # OK
        self.image_ids = []                     # OK
        self.aligned_bbs = []
        self.spatial_adj_matrices = []
        self.semantic_adj_matrices = []

        self.dataroot = dataroot

        prefix = '36'
        if 'test' in name:
            prefix = '_36'

        h5_dataroot = dataroot+"/Bottom-up-features-adaptive"\
            if self.adaptive else dataroot+"/Bottom-up-features-fixed"
        imgid_dataroot = dataroot+"/imgids"

        self.img_id2idx = pickle.load(
            open(os.path.join(imgid_dataroot, '%s%s_imgid2idx.pkl' %
                              (name, '' if self.adaptive else prefix)), 'rb'))

        h5_path = os.path.join(h5_dataroot, '%s%s.hdf5' %
                               (name, '' if self.adaptive else prefix))

        print('loading features from h5 file %s' % h5_path)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.normalized_bb = np.array(hf.get('spatial_features'))
            self.bb = np.array(hf.get('image_bb'))
            if "semantic_adj_matrix" in hf.keys() \
               and self.relation_type == "semantic":
                self.semantic_adj_matrix = np.array(
                                        hf.get('semantic_adj_matrix'))
                print("Loaded semantic adj matrix from file...",
                      self.semantic_adj_matrix.shape)
            else:
                self.semantic_adj_matrix = None
                print("Setting semantic adj matrix to None...")
            if "image_adj_matrix" in hf.keys()\
               and self.relation_type == "spatial":
                self.spatial_adj_matrix = np.array(hf.get('image_adj_matrix'))
                print("Loaded spatial adj matrix from file...",
                      self.spatial_adj_matrix.shape)
            else:
                self.spatial_adj_matrix = None
                print("Setting spatial adj matrix to None...")

            self.pos_boxes = None
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)

        self.tokenize()
        self.tensorize()
        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        if self.adaptive:
           self.v_dim = self.features.shape[1]
           self.s_dim = self.normalized_bb.shape[1]
        else:
           self.v_dim = self.features.shape[2]
           self.s_dim = self.normalized_bb.shape[2]
        
        self.num_total_data = len(self.entries)
        
        self.batch_entries = np.array_split(self.entries, len(self.entries) // batch_size)
        self.data_loader_len = len(self.batch_entries)
        

    def tokenize(self, max_length=14):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad to the back of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):

        if self.semantic_adj_matrix is not None:
            self.semantic_adj_matrix = self.semantic_adj_matrix.astype(np.float32, copy = False)
        if self.spatial_adj_matrix is not None:
            self.spatial_adj_matrix = self.spatial_adj_matrix.astype(np.float32, copy = False)

        for entry in self.entries:
            question = np.array(entry['q_token'])
            entry['q_token'] = question

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def split_entries(self, i):

        entries = self.batch_entries[i]

        features          = []
        normalized_bbs    = []
        questions         = []
        targets           = []
        bbs               = []
        spatial_adj_mats  = []
        semantic_adj_mats = []

        for entry in entries:
            entry_img     = entry['image']
            feature       = self.features[self.pos_boxes[entry_img][0]:self.pos_boxes[entry_img][1], :]
            normalized_bb = self.normalized_bb[self.pos_boxes[entry_img][0]:self.pos_boxes[entry_img][1], :]
            bb            = self.bb[self.pos_boxes[entry_img][0]:self.pos_boxes[entry_img][1], :]

            features.append(feature)
            normalized_bbs.append(normalized_bb)
            questions.append(entry['q_token'])

            bbs.append(bb)

            answer = entry['answer']

            labels, scores = answer['labels'], answer['scores']
            target = np.zeros(self.num_ans_candidates)

            if labels is not None:
                np.put_along_axis(target, labels, scores, 0)

            targets.append(target)
            spatial_adj_mats.append(np.zeros(1))
            semantic_adj_mats.append(np.zeros(1))


        return self.trim_collate(features, normalized_bbs, questions,
                                 bbs, targets, spatial_adj_mats, semantic_adj_mats)


    def trim_collate(self, features, normalized_bbs, questions, bbs, targets, spatial_adj_mats, semantic_adj_mats):

        # [bottom-up-features]
        max_len_feature = max([x.shape[0] for x in features])
        new_features = pad_sequences([feature for feature in features], padding = 'post', maxlen = max_len_feature, dtype = np.float32)

        # [normalized_bb]
        # padding -> stack
        max_len_n_bbs = max([x.shape[0] for x in normalized_bbs])
        new_n_bbs     = pad_sequences([n_bb for n_bb in normalized_bbs], padding = 'post', maxlen = max_len_n_bbs, dtype = np.float32)

        # [ question ]
        new_questions = tf.stack(questions, axis = 0)

        # bbs
        max_len_bbs = max([x.shape[0] for x in bbs])
        new_bbs     = pad_sequences([bb for bb in bbs], padding = 'post', maxlen = max_len_bbs, dtype = np.float32)


        new_targets = tf.convert_to_tensor(np.array(targets, dtype = np.float32))

        stacked_spatial  = np.stack(spatial_adj_mats, axis = 0)
        stacked_semantic = np.stack(semantic_adj_mats, axis = 0)

        return new_features, new_n_bbs, new_questions, new_targets, \
               new_bbs, stacked_spatial, stacked_semantic

    def generator(self):
        np.random.shuffle(self.entries)
        self.batch_entries = np.array_split(self.entries, len(self.entries) // self.batch_size)
        for i in range(self.data_loader_len):
            yield self.split_entries(i)

def tfidf_from_questions(names, dictionary, dataroot = './data',
                         target = ['vqa', 'vg']):
    inds = [[], []]
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1])
                inds[1].append(c[0])

    if 'vqa' in target:
        #print("[DEBUG] vqa target: ", target)
        for name in names:
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
                (name + '2014' if 'test' != name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    # Visual Genome
    if 'vg' in target:
        #print("[DEBUG] vg target: ", target)
        question_path = os.path.join(dataroot, 'visualGenome',
                                     'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])


    inds = np.load(os.path.join(dataroot, "tfidf", "indices.npy"))
    vals = np.load(os.path.join(dataroot, "tfidf", "values.npy"))
    dense_shape = (19901, 28333)
    tfidf = tf.sparse.SparseTensor(indices = inds, values = vals, dense_shape = dense_shape)

    # Latent word embeddings
    emb_dim = 300
    glove_file = dataroot+'/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(
                        dictionary.idx2word[N:], glove_file)

    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.shape[0],
          tfidf.shape[1]))

    return tfidf, weights
