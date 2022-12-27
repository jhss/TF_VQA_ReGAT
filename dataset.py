"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
This code is modified by Juhong from Linjie Li's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
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

# TODO: merge dataset_cp_v2.py with dataset.py

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
		#self.v_dim = self.features.size(1 if self.adaptive else 2)
        #self.s_dim = self.normalized_bb.size(1 if self.adaptive else 2)
        self.entries = np.array_split(self.entries, len(self.entries) // batch_size)
        self.data_len = len(self.entries)

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
            self.semantic_adj_matrix = self.semantic_adj_matrix.astype(np.float64, copy = False)
        if self.spatial_adj_matrix is not None:
            self.spatial_adj_matrix = self.spatial_adj_matrix.astype(np.double, copy = False)

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

        entries = self.entries[i]

        #entries = self.entries[:9]
        #print("[DEBUG] len entries: ", len(entries))
        features          = []
        normalized_bbs    = []
        questions         = []
        targets           = []
        question_ids      = []
        image_ids         = []
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

            question_ids.append(entry['question_id'])
            image_ids.append(entry['image_id'])
            bbs.append(bb)

            answer = entry['answer']

            labels, scores = answer['labels'], answer['scores']
            target = np.zeros(self.num_ans_candidates)

            if labels is not None:
                np.put_along_axis(target, labels, scores, 0)

            targets.append(target)
            spatial_adj_mats.append(np.zeros(1))
            semantic_adj_mats.append(np.zeros(1))


        return self.trim_collate(features, normalized_bbs, questions, question_ids,
                                 image_ids, bbs, targets, spatial_adj_mats, semantic_adj_mats)


    def trim_collate(self, features, normalized_bbs, questions, question_ids, image_ids, bbs, targets, spatial_adj_mats, semantic_adj_mats):

        # [bottom-up-features]
        max_len_feature = max([x.shape[0] for x in features])
        new_features = pad_sequences([feature for feature in features], padding = 'post', maxlen = max_len_feature, dtype = np.float32)

        #del features

        # [normalized_bb]
        # padding -> stack
        max_len_n_bbs = max([x.shape[0] for x in normalized_bbs])
        new_n_bbs     = pad_sequences([n_bb for n_bb in normalized_bbs], padding = 'post', maxlen = max_len_n_bbs, dtype = np.float32)
        #del normalized_bbs

        # [ question ]
        new_questions = tf.stack(questions, axis = 0)

        # question_ids
        new_question_ids = np.array(question_ids)

        # image_ids
        new_image_ids = np.array(image_ids)

        # bbs
        max_len_bbs = max([x.shape[0] for x in bbs])
        new_bbs     = pad_sequences([bb for bb in bbs], padding = 'post', maxlen = max_len_bbs, dtype = np.float32)


        new_targets = tf.convert_to_tensor(np.array(targets, dtype = np.float32))

        stacked_spatial  = np.stack(spatial_adj_mats, axis = 0)
        stacked_semantic = np.stack(semantic_adj_mats, axis = 0)

        return new_features, new_n_bbs, new_questions, new_targets, new_question_ids,\
               new_image_ids, new_bbs, stacked_spatial, stacked_semantic

    def generator(self):
        for i in range(self.data_len):
            yield self.split_entries(i)


class VQAFeatureDataset2(Sequence):
    def __init__(self, name, dictionary, relation_type, batch_size, dataroot='data',
                 adaptive=False, pos_emb_dim=64, nongt_dim=36):
        super(VQAFeatureDataset2, self).__init__()

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
        print("[DEBUG] self.features: ", type(self.features))
        #print("[DEBUG] self.features: ", self.features)
        if self.adaptive:
            self.v_dim = self.features.shape[1]
            self.s_dim = self.normalized_bb.shape[1]
        else:
            self.v_dim = self.features.shape[2]
            self.s_dim = self.normalized_bb.shape[2]
        #self.v_dim = self.features.size(1 if self.adaptive else 2)
        #self.s_dim = self.normalized_bb.size(1 if self.adaptive else 2)
        print("[DEBUG] self.v_dim: ", self.v_dim, " self.s_dim: ", self.s_dim)

        self.entries = np.array_split(self.entries, len(self.entries) // batch_size)
        self.data_len = len(self.entries)
        self.train_entries, self.val_entries = train_test_split(self.entries, test_size = 0.2)
        self.train_data_len = len(self.train_entries)
        self.val_data_len   = len(self.val_entries)

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
            self.semantic_adj_matrix = self.semantic_adj_matrix.astype(np.float64, copy = False)
        if self.spatial_adj_matrix is not None:
            self.spatial_adj_matrix = self.spatial_adj_matrix.astype(np.double, copy = False)

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

    def split_entries(self, i, mode):

        if mode == 'train':
            entries = self.train_entries[i]
        else:
            entries = self.val_entries[i]
            #print("len enries: ", len(entries))
            #print("entries[0]: ", entries[0])

        features          = []
        normalized_bbs    = []
        questions         = []
        targets           = []
        question_ids      = []
        image_ids         = []
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

            question_ids.append(entry['question_id'])
            image_ids.append(entry['image_id'])
            bbs.append(bb)

            answer = entry['answer']

            labels, scores = answer['labels'], answer['scores']
            target = np.zeros(self.num_ans_candidates)

            if labels is not None:
                np.put_along_axis(target, labels, scores, 0)

            targets.append(target)
            spatial_adj_mats.append(np.zeros(1))
            semantic_adj_mats.append(np.zeros(1))


        return self.trim_collate(features, normalized_bbs, questions, question_ids,
                                 image_ids, bbs, targets, spatial_adj_mats, semantic_adj_mats)


    def trim_collate(self, features, normalized_bbs, questions, question_ids, image_ids, bbs, targets, spatial_adj_mats, semantic_adj_mats):

        # [bottom-up-features]
        max_len_feature = max([x.shape[0] for x in features])
        new_features = pad_sequences([feature for feature in features], padding = 'post', maxlen = max_len_feature, dtype = np.float32)

        #del features

        # [normalized_bb]
        # padding -> stack
        max_len_n_bbs = max([x.shape[0] for x in normalized_bbs])
        new_n_bbs     = pad_sequences([n_bb for n_bb in normalized_bbs], padding = 'post', maxlen = max_len_n_bbs, dtype = np.float32)
        #del normalized_bbs

        # [ question ]
        new_questions = tf.stack(questions, axis = 0)
        #del questions
        #print("[DEBUG] questions.shape: ", new_questions.shape)

        # question_ids
        new_question_ids = np.array(question_ids)
        #print("[DEBUG] question_ids.shape: ", np.array(question_ids).shape)
        #del question_ids
        #print("[DEBUG] question_ids type: ", type(question_ids))

        # image_ids
        new_image_ids = np.array(image_ids)
        #print("[DEBUG] image_ids: ", self.image_ids[0].shape)
        #del image_ids

        # bbs
        max_len_bbs = max([x.shape[0] for x in bbs])
        new_bbs     = pad_sequences([bb for bb in bbs], padding = 'post', maxlen = max_len_bbs, dtype = np.float32)

        #print("[DEBUG] bbs.shape: ", new_bbs.shape)
        #del bbs

        # targets
        #print("[DEBUG] targets type: ", type(targets))
        #print("[DEBUG] targets[0].type: ", type(targets[0]))
        new_targets = tf.convert_to_tensor(np.array(targets, dtype = np.float32))
        #self.targets.append(tf.convert_to_tensor(np_targets))
        #print("[DEBUG] targets.shape: ", self.targets[0].shape)
        #del targets, np_targets

        stacked_spatial  = np.stack(spatial_adj_mats, axis = 0)
        stacked_semantic = np.stack(semantic_adj_mats, axis = 0)

        return new_features, new_n_bbs, new_questions, new_targets, new_question_ids,\
               new_image_ids, new_bbs, stacked_spatial, stacked_semantic


        #print("[DEBUG] stacked_spatial.shape: ", stacked_spatial.shape)

        #np.save(self.dataroot + "features.npy", self.features[0])
        #np.save(self.dataroot + "n_bb.npy", self.aligned_normalized_bbs[0])
        #np.save(self.dataroot + "question.npy", self.questions[0])
        #np.save(self.dataroot + "question_ids.npy", self.question_ids[0])
        #np.save(self.dataroot + "image_ids.npy", self.image_ids[0])
        #np.save(self.dataroot + "bbs.npy", self.aligned_bbs[0])
        #np.save(self.dataroot + "targets.npy", self.targets[0])
        #print("[DEBUG] aligned_features[0]: ", self.aligned_features[0])
        #print("[DEBUG] aligned_features[0] dtype: ", self.aligned_features[0].dtype)
        # [ CHECK ] numpy
    def train_generator(self):
        #return lambda: (self.aligned_features[i] for i in range(self.data_len))
        for i in range(self.train_data_len):
            #print("[DEBUG] idx: ", i, " aligned_Features length: ", len(self.aligned_features))
            yield self.split_entries(i, 'train')

    def val_generator(self):

        for i in range(self.val_data_len):
            yield self.split_entries(i, 'val')

    def __getitem__(self, index):

        start_idx = index * self.batch_size
        end_idx   = (index+1) * self.batch_size

        return self.aligned_features[start_idx:end_idx], self.aligned_normalized_bbs[start_idx:end_idx],\
               self.questions[start_idx:end_idx], self.targets[start_idx:end_idx],\
                self.question_ids[start_idx:end_idx], self.image_ids[start_idx:end_idx], self.aligned_bb[start_idx:end_idx],


    def __len__(self):
        return math.ceil(len(self.entries) / self.batch_size)


def tfidf_from_questions(names, dictionary, dataroot = '../VQA_ReGAT/data',
                         target = ['vqa', 'vg']):
    inds = [[], []]
    df = dict()
    N = len(dictionary)
    print("[DEBUG] N: ", N)

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
    print("[DEBUG] vqa dic length: ", len(dictionary))
    print("len inds: ", len(inds))
    print("len inds[0]: ", len(inds[0]), " len inds[1]: ", len(inds[1]))
    # Visual Genome
    if 'vg' in target:
        #print("[DEBUG] vg target: ", target)
        question_path = os.path.join(dataroot, 'visualGenome',
                                     'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    print("[DEBUG] vg dic length: ", len(dictionary))



    inds = np.load(os.path.join(dataroot, "tfidf", "indices.npy"))
    vals = np.load(os.path.join(dataroot, "tfidf", "values.npy"))
    dense_shape = (19901, 28333)
    tfidf = tf.sparse.SparseTensor(indices = inds, values = vals, dense_shape = dense_shape)
    #print("[DEBUG] tfidf: ", tfidf)

    # Latent word embeddings
    emb_dim = 300
    glove_file = dataroot+'/glove/glove.6B.%dd.txt' % emb_dim
    #print("[DEBUG] dictionary.idx2word inside func: ", dictionary.idx2word)
    print("[DEBUG] dictionary len: ", N)
    print("[DEBUG] dictionary.idx2word len: ", len(dictionary.idx2word))
    weights, word2emb = utils.create_glove_embedding_init(
                        dictionary.idx2word[N:], glove_file)

    #print("[DEBUG] utils.create_glove_weights: ", weights)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.shape[0],
          tfidf.shape[1]))

    return tfidf, weights
