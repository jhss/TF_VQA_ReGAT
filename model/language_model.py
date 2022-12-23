import numpy as np
import tensorflow as tf

from model.fc import FullyConnected
"""
Embedding class is modifed by Juhong from Sungwoo kyoo's blog.
https://sungwookyoo.github.io/tips/CompareTensorflowAndPytorch/#1-tensorflow
"""

class Embedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, padding_idx = 0, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.padding_idx = padding_idx

        self.embeddings = self.add_weight(
                            shape = (input_dim, output_dim),
                            initializer = 'random_normal',
                            dtype = 'float32'
                            )

    def call(self, inputs):
        """
        Inputs:
            inputs: [batch, seq_len]

        Returns:
            masked_embes: [batch, seq_len, emb_dim]
        """

        embeds = tf.nn.embedding_lookup(self.embeddings, inputs)
        mask   = tf.not_equal(inputs, self.padding_idx)
        mask   = tf.tile(mask[:, :, tf.newaxis], [1, 1, self.output_dim])
        mask   = tf.cast(mask, dtype = tf.float32)

        masked_embeds = embeds * mask

        return masked_embeds

"""
The remaining code is modifed by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Licensed under the MIT license.
"""

class WordEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_token, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op  = op
        self.emb = Embedding(n_token+1, emb_dim, padding_idx = n_token)

        if 'c' in op:
            self.emb_ = Embedding(n_token+1, emb_dim, padding_idx = n_token)
            self.emb_.trainable = False

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.n_token = n_token
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tf_idf = None, tf_idf_weights = None):
        np_weight = np.load(np_file)
        weight_init = tf.squeeze(tf.convert_to_tensor(np_weight))
        del np_weight
        assert weight_init.shape == (self.n_token, self.emb_dim)

        pads        = tf.zeros(shape = (1, self.emb_dim), dtype = tf.float32)
        print("[DEUBG] weight_init.shape: ", weight_init.shape)
        #new_weights = tf.expand_dims(tf.concat([weight_init, pads], axis = 0), axis = 0)
        new_weights = tf.concat([weight_init, pads], axis = 0)
        #print("[DEBUG} new_weights.shape: ", new_weights.shape)
        print("[DEBUG] get_weights.shape: ", self.emb.get_weights()[0].shape)
        print("[DEBUG] get_weights length: ", len(self.emb.get_weights()))
        print("[DEBUG] get_weights length: ", type(self.emb.get_weights()))
        #print("[DEBUG] get_weights shape: ", self.emb.get_weights().shape)
        print("[DE]")
        self.emb.build((None,))
        self.emb.set_weights([new_weights])
        print("[DEBUG] CLEAR")
        if tf_idf is not None:

            print("[DEBUG] Before concat: ", weight_init.shape)
            tensor_tf_idf_weights = tf.convert_to_tensor(tf_idf_weights)
            print("[DEBUG] tf_idf_weights: ", tf_idf_weights)
            del tf_idf_weights
            print("[DEBUG] tf_weights.shape: ", tensor_tf_idf_weights.shape)
            weight_init = tf.concat([weight_init,
                                     tensor_tf_idf_weights], axis = 0)

            print("[DEBUG] sparse: ", tf_idf)
            print("[DEBUG] After concat weight_init.shape: ", weight_init.shape)
            weight_init = tf.sparse.sparse_dense_matmul(tf_idf, weight_init)
            #weight_init = tf.matmul(tf_idf, weight_init)

            if 'c' in self.op:
                self.emb_.trainable = True

        if 'c' in self.op:
            new_weights = tf.concat([tf.identity(weight_init),
                                     tf.identity(pads)], axis = 0)
            self.emb_.build((None,))
            self.emb_.set_weights([new_weights])

    def call(self, input):
        emb = self.emb(input)
        if 'c' in self.op:
            emb = tf.concat([emb, self.emb_(input)], axis = 2)
        emb = self.dropout(emb)

        return emb

class QuestionEmbedding(tf.keras.layers.Layer):
    def __init__(self, in_dim, hidden_dim, n_layers, bidirect, dropout, rnn_type = 'GRU'):
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'

        self.gru          = tf.keras.layers.GRU(units = hidden_dim, dropout = dropout,
                                                return_sequences = True,
                                                return_state = True)
        self.in_dim       = in_dim
        self.hidden_dim   = hidden_dim

        self.n_layers     = n_layers
        self.rnn_type     = rnn_type
        self.n_directions = 1 + int(bidirect)

    #def init_hidden(self, batch):
    def call_last(self, question):

        output, hidden = self.gru(question)

        return output[:, -1]
    # [Note] Replace 'forward_all' in the PyTorch code with 'call', because 'forward' seems not to be used in the code.
    def call(self, input):

        #print("[DEBUG] question emb input.shape: ", input.shape)
        batch  = input.shape[0]
        #initial_hidden = self.init_hidden(batch)
        output, hidden = self.gru(input) # [CHECK] Pytorch Code 실행할 때 initial_hidden_state 없애고 실행해도 결과값 같은지 (예쌍: 같다)

        #print("[DEBUG] output.shape: ", output.shape, " hidden.shape: ", hidden.shape)
        return output


class QuestionSelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, dropout):
        super(QuestionSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = tf.keras.layers.Dropout(dropout)
        self.linear1    = FullyConnected(dims = [hidden_dim, hidden_dim], dropout = dropout,
                                         activation = None)
        self.act        = tf.keras.layers.Activation('tanh')
        self.linear2    = FullyConnected(dims = [hidden_dim, 1], activation = None)

    def call(self, question):
        """
        question: [batch, 14, hidden_dim] -> what 14 means?
        """

        #print("[DEBUG] question.shape: ", question.shape)
        # [DEBUG] question.shape:  (9, 1024)
        batch, seq_len = question.shape[0], question.shape[1]
        reshaped_question = tf.reshape(question, shape = (-1, self.hidden_dim))

        logits  = self.linear2(self.act(self.linear1(reshaped_question)))
        #print("[DEBUG] logits.shape: ", logits.shape)
        # [DEBUG] logits.shape:  (9, 1)

        logits  = tf.reshape(logits, (batch, seq_len))
        # [batch, 1, 14]
        attention_weights = tf.reshape(tf.nn.softmax(logits, axis = 1), # 원본에서는 transpose사용했는데, 잘 모르겠음
                                       shape = (-1, 1, seq_len))

        # attention_weight: [9, 1, 14], question: [9, 1024]
        self_att_question = tf.reshape(tf.matmul(attention_weights, question),
                                       shape = (-1, self.hidden_dim))

        self_att_question = self.dropout(self_att_question)

        return self_att_question
