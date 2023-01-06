import numpy as np
import tensorflow as tf
import numpy as np
from model.fc import FullyConnected
"""
Embedding class is modifed by Juhong from Sungwoo kyoo's blog.
https://sungwookyoo.github.io/tips/CompareTensorflowAndPytorch/#1-tensorflow
"""

class Embedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, name, padding_idx = 0, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.padding_idx = padding_idx

        self.embeddings = self.add_weight(
                            shape = (input_dim, output_dim),
                            initializer = 'random_normal',
                            name = name,
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
        self.emb = Embedding(n_token+1, emb_dim, name = 'emb', padding_idx = n_token)

        if 'c' in op:
            self.emb_ = Embedding(n_token+1, emb_dim, name = 'emb_', padding_idx = n_token)
            self.emb_.trainable = False

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.n_token = n_token
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tf_idf = None, tf_idf_weights = None):
        np_weight = np.load(np_file)
        weight_init = tf.squeeze(tf.convert_to_tensor(np_weight, dtype = tf.float32))
        del np_weight
        assert weight_init.shape == (self.n_token, self.emb_dim)

        pads        = tf.zeros(shape = (1, self.emb_dim), dtype = tf.float32)
        new_weights = tf.concat([weight_init, pads], axis = 0)
        self.emb.build((None,))
        self.emb.set_weights([new_weights])

        if tf_idf is not None:

            tensor_tf_idf_weights = tf.convert_to_tensor(tf_idf_weights, dtype = tf.float32)
            del tf_idf_weights
            weight_init = tf.concat([weight_init,
                                     tensor_tf_idf_weights], axis = 0)

            weight_init = tf.sparse.sparse_dense_matmul(tf_idf, weight_init)

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

        dropout = 0.0
        self.gru          = tf.keras.layers.GRU(units = hidden_dim, dropout = dropout,
                                                return_sequences = True,
                                                return_state = True)
        self.in_dim       = in_dim
        self.hidden_dim   = hidden_dim

        self.n_layers     = n_layers
        self.rnn_type     = rnn_type
        self.n_directions = 1 + int(bidirect)

    def call_last(self, question):
        """
        Return only last hidden states
        """
        output, hidden = self.gru(question)

        return output[:, -1]

    def call(self, input):
        """
        Return all hidden states
         """

        batch  = input.shape[0]
        output, hidden = self.gru(input) 

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
        Inputs:
            question: [batch, 14, hidden_dim]
        
        Returns:
            self_att_question: [batch, hidden_dim]
        """

        batch, seq_len = question.shape[0], question.shape[1]

        atten_1 = self.linear1(question)

        atten_1 = self.act(atten_1)

        logits  = self.linear2(atten_1)

        logits  = tf.squeeze(logits)

        # [batch, 1, 14]
        attention_weights = tf.reshape(tf.nn.softmax(tf.transpose(logits), axis = 1),
                                       shape = (batch, 1, seq_len))

        self_att_question = tf.reshape(tf.matmul(attention_weights, question), 
                                       shape = (batch, self.hidden_dim))

        self_att_question = self.dropout(self_att_question)

        return self_att_question
