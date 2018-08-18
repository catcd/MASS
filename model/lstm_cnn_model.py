import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.rnn import BasicLSTMCell

from dataset.dataset import pad_sequences
from utils import Timer, Log

seed = 13
np.random.seed(seed)


class LstmCnn:
    def __init__(self, model_name, embeddings_fasttext, embeddings_wordnet_superset, batch_size, constants):
        self.model_name = model_name
        self.embeddings_fasttext = embeddings_fasttext
        self.embeddings_wordnet_superset = embeddings_wordnet_superset
        self.batch_size = batch_size
        self.max_sent_length = constants.MAX_SENT_LENGTH

        self.use_cnn_region_size_1 = constants.USE_CNN_REGION_SIZE_1
        self.cnn_region_size_1_filter = constants.CNN_REGION_SIZE_1_FILTER
        self.use_cnn_region_size_2 = constants.USE_CNN_REGION_SIZE_2
        self.cnn_region_size_2_filter = constants.CNN_REGION_SIZE_2_FILTER
        self.use_cnn_region_size_3 = constants.USE_CNN_REGION_SIZE_3
        self.cnn_region_size_3_filter = constants.CNN_REGION_SIZE_3_FILTER

        if not self.use_cnn_region_size_1 and not self.use_cnn_region_size_2 and not self.use_cnn_region_size_3:
            raise ValueError('Must use at least 1 region size for CNN')

        self.use_char = constants.USE_CHAR
        self.nchars = constants.NCHARS
        self.input_lstm_char_dim = constants.INPUT_LSTM_CHAR_DIM
        self.output_lstm_char_dim = constants.OUTPUT_LSTM_CHAR_DIM

        self.use_pos = constants.USE_POS
        self.npos = constants.NPOS
        self.input_lstm_pos_dim = constants.INPUT_LSTM_POS_DIM
        self.output_lstm_pos_dim = constants.OUTPUT_LSTM_POS_DIM

        self.use_relation = constants.USE_RELATION
        self.nrelations = constants.NRELATIONS
        self.input_lstm_relation_dim = constants.INPUT_LSTM_RELATION_DIM
        self.output_lstm_relation_dim = constants.OUTPUT_LSTM_RELATION_DIM

        self.use_direction = constants.USE_DIRECTION
        self.ndirections = constants.NDIRECTIONS
        self.direction_embedding_dim = constants.DIRECTION_EMBEDDING_DIM

        self.use_fasttext = constants.USE_FASTTEXT
        self.input_fasttext_dim = constants.INPUT_FASTTEXT_DIM
        self.output_lstm_fasttext_dim = constants.OUTPUT_LSTM_FASTTEXT_DIM

        self.use_wordnet_superset = constants.USE_WORDNET_SUPERSET
        self.input_wordnet_superset_dim = constants.INPUT_WORDNET_SUPERSET_DIM
        self.output_lstm_wordnet_superset_dim = constants.OUTPUT_LSTM_WORDNET_SUPERSET_DIM

        self.hidden_layers = constants.HIDDEN_LAYERS

        self.alpha = constants.ALPHA

        self.use_word = (
                self.use_char
                or self.use_pos
                or self.use_fasttext
                or self.use_wordnet_superset
        )
        self.use_dependency = self.use_relation
        if not self.use_word and not self.use_dependency:
            raise ValueError('Must use at least 1 chanel for word embedding or use dependency embedding')

        self.num_of_class = len(constants.ALL_LABELS)
        self.all_labels = constants.ALL_LABELS
        self.num_of_class_2 = len(constants.ALL_LABELS_2)
        self.all_labels_2 = constants.ALL_LABELS_2

        self.trained_models = constants.TRAINED_MODELS

        self.LABEL_2_LABEL2_MAP = constants.LABEL_2_LABEL2_MAP
        self.LABEL_2_LABELB_MAP = constants.LABEL_2_LABELB_MAP

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.labels = tf.placeholder(name="y_true", shape=[None], dtype='int32')
        self.word_ids = tf.placeholder(name='word_ids', shape=[None, None], dtype='int32')
        self.wordnet_superset_ids = tf.placeholder(name='wordnet_superset_ids', shape=[None, None], dtype='int32')
        self.char_ids = tf.placeholder(name='char_ids', shape=[None, None, None], dtype='int32')
        self.word_lengths = tf.placeholder(name="word_lengths", shape=[None, None], dtype='int32')
        self.word_pos_ids = tf.placeholder(name='word_pos', shape=[None, None], dtype='int32')
        self.relation = tf.placeholder(name='relation', dtype=tf.int32, shape=[None, None])
        self.direction = tf.placeholder(name='direction', dtype=tf.int32, shape=[None, None])
        self.sequence_lens = tf.placeholder(name='sequence_lens', dtype=tf.int32, shape=[None])
        self.relation_lens = tf.placeholder(name='relation_lens', dtype=tf.int32, shape=[None])
        self.labels_2 = tf.placeholder(name="y_true_2", shape=[None], dtype='int32')
        self.dropout_op = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_op")
        self.dropout_final_embedding = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_final_embedding')
        self.dropout_dependency = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_dependency')
        self.dropout_cnn = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_cnn')
        self.labels_b = tf.placeholder(name="y_true_b", shape=[None], dtype='int32')
        self.is_training = tf.placeholder(tf.bool, name='phase')
        self.is_final = tf.placeholder(tf.bool, name='phase2')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        if self.use_fasttext:
            with tf.variable_scope("fasttext_embedding"):
                _fasttext_embeddings = tf.Variable(self.embeddings_fasttext, name="lut", dtype=tf.float32, trainable=False)
                self.fasttext_embeddings = tf.nn.embedding_lookup(
                    _fasttext_embeddings, self.word_ids,
                    name="embeddings"
                )
                self.fasttext_embeddings = tf.nn.dropout(self.fasttext_embeddings, self.dropout_op)

            with tf.variable_scope("bi_lstm_fasttext"):
                cell_fw = BasicLSTMCell(self.output_lstm_fasttext_dim)
                cell_bw = BasicLSTMCell(self.output_lstm_fasttext_dim)

                (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    self.fasttext_embeddings,
                    sequence_length=self.sequence_lens,
                    dtype=tf.float32
                )
            with tf.variable_scope("bi_lstm_fasttext_f"):
                self.output_lstm_fasttext_w2v_f = self.batch_normalization(fw_output, training=self.is_training)
            with tf.variable_scope("bi_lstm_fasttext_b"):
                self.output_lstm_fasttext_w2v_b = self.batch_normalization(bw_output, training=self.is_training)

        if self.use_wordnet_superset:
            with tf.variable_scope("wordnet_superset_embedding"):
                _wordnet_superset_embeddings = tf.Variable(
                    self.embeddings_wordnet_superset,
                    name="lut", dtype=tf.float32, trainable=False
                )
                self.wordnet_superset_embeddings = tf.nn.embedding_lookup(
                    _wordnet_superset_embeddings,
                    self.wordnet_superset_ids,
                    name="embeddings"
                )
                self.wordnet_superset_embeddings = tf.nn.dropout(self.wordnet_superset_embeddings, self.dropout_op)

            with tf.variable_scope("bi_lstm_wordnet_superset"):
                cell_fw = BasicLSTMCell(self.output_lstm_wordnet_superset_dim)
                cell_bw = BasicLSTMCell(self.output_lstm_wordnet_superset_dim)
                (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    self.wordnet_superset_embeddings,
                    sequence_length=self.sequence_lens,
                    dtype=tf.float32
                )
            with tf.variable_scope("bi_lstm_wordnet_superset_f"):
                self.output_lstm_wordnet_superset_f = self.batch_normalization(fw_output, training=self.is_training)
            with tf.variable_scope("bi_lstm_wordnet_superset_b"):
                self.output_lstm_wordnet_superset_b = self.batch_normalization(bw_output, training=self.is_training)

        if self.use_relation:
            with tf.variable_scope('relation_embedding'):
                _relation_embeddings = tf.get_variable(
                    name='lut', dtype=tf.float32,
                    shape=[self.nrelations, self.input_lstm_relation_dim],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )
                relation_embeddings = tf.nn.embedding_lookup(
                    _relation_embeddings, self.relation,
                    name='embeddings'
                )

                if self.use_direction:
                    with tf.variable_scope('direction'):
                        _direction_embeddings = tf.get_variable(
                            name='lut', dtype=tf.float32,
                            shape=[self.ndirections, self.direction_embedding_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                        )
                        direction_embeddings = tf.nn.embedding_lookup(
                            _direction_embeddings, self.direction,
                            name='embeddings'
                        )
                        relation_embeddings = tf.concat([relation_embeddings, direction_embeddings], axis=-1)

                relation_embeddings = tf.nn.dropout(relation_embeddings, self.dropout_op)

                with tf.variable_scope("bi_lstm_dependency"):
                    cell_fw = BasicLSTMCell(self.output_lstm_relation_dim)
                    cell_bw = BasicLSTMCell(self.output_lstm_relation_dim)
                    (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw,
                        relation_embeddings,
                        sequence_length=self.relation_lens,
                        dtype=tf.float32
                    )
                with tf.variable_scope("bi_lstm_dependency_f"):
                    self.output_lstm_relation_f = self.batch_normalization(fw_output, training=self.is_training)
                with tf.variable_scope("bi_lstm_dependency_b"):
                    self.output_lstm_relation_b = self.batch_normalization(bw_output, training=self.is_training)

        if self.use_pos:
            with tf.variable_scope('pos_embedding'):
                _pos_embeddings = tf.get_variable(
                    name='lut', dtype=tf.float32,
                    shape=[self.npos, self.input_lstm_pos_dim],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )
                pos_embeddings = tf.nn.embedding_lookup(
                    _pos_embeddings, self.word_pos_ids,
                    name='embeddings'
                )
                pos_embeddings = tf.nn.dropout(pos_embeddings, self.dropout_op)

            with tf.variable_scope("bi_lstm_pos"):
                cell_fw = BasicLSTMCell(self.output_lstm_pos_dim)
                cell_bw = BasicLSTMCell(self.output_lstm_pos_dim)
                (fw_output_pos, bw_output_pos), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    pos_embeddings,
                    sequence_length=self.sequence_lens,
                    dtype=tf.float32
                )

            with tf.variable_scope("bi_lstm_pos_f"):
                self.output_pos_f = self.batch_normalization(fw_output_pos, training=self.is_training)

            with tf.variable_scope("bi_lstm_pos_b"):
                self.output_pos_b = self.batch_normalization(bw_output_pos, training=self.is_training)

        if self.use_char:
            with tf.variable_scope("chars_embedding"):
                _char_embeddings = tf.get_variable(
                    name="lut", dtype=tf.float32,
                    shape=[self.nchars, self.input_lstm_char_dim],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="embeddings")
                char_embeddings = tf.nn.dropout(char_embeddings, self.dropout_op)

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.input_lstm_char_dim])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])

            with tf.variable_scope("bi_lstm_char"):
                cell_fw = BasicLSTMCell(self.output_lstm_char_dim)
                cell_bw = BasicLSTMCell(self.output_lstm_char_dim)
                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    char_embeddings,
                    sequence_length=word_lengths,
                    dtype=tf.float32
                )

            with tf.variable_scope("bi_lstm_char_output"):
                self.output_char_f = tf.reshape(output_fw, shape=[-1, s[1], self.output_lstm_char_dim])
                self.output_char_b = tf.reshape(output_bw, shape=[-1, s[1], self.output_lstm_char_dim])

                self.output_char = tf.concat([self.output_char_f, self.output_char_b], axis=-1)

    @staticmethod
    def batch_normalization(inputs, training, decay=0.9, epsilon=1e-3):

        scale = tf.get_variable('scale', inputs.get_shape()[-1], initializer=tf.ones_initializer(), dtype=tf.float32)
        beta = tf.get_variable('beta', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        pop_mean = tf.get_variable('pop_mean', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                                   dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable('pop_var', inputs.get_shape()[-1], initializer=tf.ones_initializer(),
                                  dtype=tf.float32, trainable=False)

        axis = list(range(len(inputs.get_shape()) - 1))

        def Train():
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            pop_mean_new = pop_mean * decay + batch_mean * (1 - decay)
            pop_var_new = pop_var * decay + batch_var * (1 - decay)
            with tf.control_dependencies([pop_mean.assign(pop_mean_new), pop_var.assign(pop_var_new)]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

        def Eval():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        return tf.cond(training, Train, Eval)

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        if self.use_word:
            embeddings_f = []
            embeddings_b = []

            if self.use_fasttext:
                embeddings_f.append(self.output_lstm_fasttext_w2v_f)
                embeddings_b.append(self.output_lstm_fasttext_w2v_b)

            if self.use_wordnet_superset:
                embeddings_f.append(self.output_lstm_wordnet_superset_f)
                embeddings_b.append(self.output_lstm_wordnet_superset_b)

            if self.use_pos:
                embeddings_f.append(self.output_pos_f)
                embeddings_b.append(self.output_pos_b)

            if self.use_char:
                embeddings_f.append(self.output_char)
                embeddings_b.append(self.output_char)

            with tf.variable_scope('all_word_embedding_f'):
                word_embedding_final_f = tf.concat(embeddings_f, axis=-1)
                word_embedding_final_f = tf.nn.dropout(word_embedding_final_f, self.dropout_final_embedding)
            with tf.variable_scope('all_word_embedding_b'):
                word_embedding_final_b = tf.concat(embeddings_b, axis=-1)
                word_embedding_final_b = tf.nn.dropout(word_embedding_final_b, self.dropout_final_embedding)
        else:
            with tf.variable_scope('all_word_embedding_f'):
                word_embedding_final_f = np.array([[[]]])
            with tf.variable_scope('all_word_embedding_b'):
                word_embedding_final_b = np.array([[[]]])

        if self.use_relation:
            with tf.variable_scope('dependency_embedding_f'):
                dependency_embedding_f = self.output_lstm_relation_f
                dependency_embedding_f = tf.nn.dropout(dependency_embedding_f, self.dropout_dependency)
            with tf.variable_scope('dependency_embedding_b'):
                dependency_embedding_b = self.output_lstm_relation_b
                dependency_embedding_b = tf.nn.dropout(dependency_embedding_b, self.dropout_dependency)
        else:
            with tf.variable_scope('dependency_embedding_f'):
                dependency_embedding_f = np.array([[[]]])
            with tf.variable_scope('dependency_embedding_b'):
                dependency_embedding_b = np.array([[[]]])

        dim_word_final = (
                (self.output_lstm_fasttext_dim if self.use_fasttext else 0)
                + (self.output_lstm_wordnet_superset_dim if self.use_wordnet_superset else 0)
                + (self.output_lstm_pos_dim if self.use_pos else 0)
                + (self.output_lstm_char_dim if self.use_char else 0) * 2
        )
        dim_dependency_final = (
            (self.output_lstm_relation_dim if self.use_relation else 0)
        )
        cnn_filter_width = 2 * dim_word_final + dim_dependency_final

        with tf.variable_scope("final_matrix_f"):
            cnn_word_rela_f = []
            if self.use_word:
                word_embedding_l = word_embedding_final_f[:, :-1, :]
                cnn_word_rela_f.append(word_embedding_l)

            if self.use_relation:
                cnn_word_rela_f.append(dependency_embedding_f)

            if self.use_word:
                word_embedding_r = word_embedding_final_f[:, 1:, :]
                cnn_word_rela_f.append(word_embedding_r)

            self.word_rela_f = tf.expand_dims(
                tf.concat(cnn_word_rela_f, axis=2), -1)

        with tf.variable_scope("conv_f"):
            conv_f = []
            if self.use_cnn_region_size_1:
                with tf.variable_scope("conv_f_1"):
                    conv_f_1 = tf.layers.conv2d(
                        self.word_rela_f, filters=self.cnn_region_size_1_filter,
                        kernel_size=(1, cnn_filter_width),
                        use_bias=False, padding="valid",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    conv_f_1 = self.batch_normalization(conv_f_1, training=self.is_training)
                    conv_f_1 = tf.reduce_max(tf.nn.tanh(conv_f_1), 1)
                    conv_f_1 = tf.reshape(conv_f_1, [-1, self.cnn_region_size_1_filter])

                    conv_f.append(conv_f_1)

            if self.use_cnn_region_size_2:
                with tf.variable_scope("conv_f_2"):
                    conv_f_2 = tf.layers.conv2d(
                        self.word_rela_f, filters=self.cnn_region_size_2_filter,
                        kernel_size=(2, cnn_filter_width),
                        use_bias=False, padding="valid",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    conv_f_2 = self.batch_normalization(conv_f_2, training=self.is_training)
                    conv_f_2 = tf.reduce_max(tf.nn.tanh(conv_f_2), 1)
                    conv_f_2 = tf.reshape(conv_f_2, [-1, self.cnn_region_size_2_filter])

                    conv_f.append(conv_f_2)

            if self.use_cnn_region_size_3:
                with tf.variable_scope("conv_f_3"):
                    conv_f_3 = tf.layers.conv2d(
                        self.word_rela_f, filters=self.cnn_region_size_3_filter,
                        kernel_size=(3, cnn_filter_width),
                        use_bias=False, padding="valid",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    conv_f_3 = self.batch_normalization(conv_f_3, training=self.is_training)
                    conv_f_3 = tf.reduce_max(tf.nn.tanh(conv_f_3), 1)
                    conv_f_3 = tf.reshape(conv_f_3, [-1, self.cnn_region_size_3_filter])

                    conv_f.append(conv_f_3)

            final_conv_f = tf.concat(conv_f, axis=-1)
            final_conv_f = tf.nn.dropout(final_conv_f, self.dropout_cnn)

        with tf.variable_scope("final_matrix_b"):
            cnn_word_rela_b = []
            if self.use_word:
                word_embedding_l = word_embedding_final_b[:, :-1, :]
                cnn_word_rela_b.append(word_embedding_l)

            if self.use_relation:
                cnn_word_rela_b.append(dependency_embedding_b)

            if self.use_word:
                word_embedding_r = word_embedding_final_b[:, 1:, :]
                cnn_word_rela_b.append(word_embedding_r)

            self.word_rela_b = tf.expand_dims(
                tf.concat(cnn_word_rela_b, axis=2), -1)

        with tf.variable_scope("conv_b"):
            conv_b = []
            if self.use_cnn_region_size_1:
                with tf.variable_scope("conv_b_1"):
                    conv_b_1 = tf.layers.conv2d(
                        self.word_rela_b, filters=self.cnn_region_size_1_filter,
                        kernel_size=(1, cnn_filter_width),
                        use_bias=False, padding="valid",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                    conv_b_1 = self.batch_normalization(conv_b_1, training=self.is_training)
                    conv_b_1 = tf.reduce_max(tf.nn.tanh(conv_b_1), 1)
                    conv_b_1 = tf.reshape(conv_b_1, [-1, self.cnn_region_size_1_filter])

                    conv_b.append(conv_b_1)

            if self.use_cnn_region_size_2:
                with tf.variable_scope("conv_b_2"):
                    conv_b_2 = tf.layers.conv2d(
                        self.word_rela_b, filters=self.cnn_region_size_2_filter,
                        kernel_size=(2, cnn_filter_width),
                        use_bias=False, padding="valid",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                    conv_b_2 = self.batch_normalization(conv_b_2, training=self.is_training)
                    conv_b_2 = tf.reduce_max(tf.nn.tanh(conv_b_2), 1)
                    conv_b_2 = tf.reshape(conv_b_2, [-1, self.cnn_region_size_2_filter])

                    conv_b.append(conv_b_2)

            if self.use_cnn_region_size_3:
                with tf.variable_scope("conv_b_3"):
                    conv_b_3 = tf.layers.conv2d(
                        self.word_rela_b, filters=self.cnn_region_size_3_filter,
                        kernel_size=(3, cnn_filter_width),
                        use_bias=False, padding="valid",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                    conv_b_3 = self.batch_normalization(conv_b_3, training=self.is_training)
                    conv_b_3 = tf.reduce_max(tf.nn.tanh(conv_b_3), 1)
                    conv_b_3 = tf.reshape(conv_b_3, [-1, self.cnn_region_size_3_filter])

                    conv_b.append(conv_b_3)

            final_conv_b = tf.concat(conv_b, axis=-1)
            final_conv_b = tf.nn.dropout(final_conv_b, self.dropout_cnn)

        with tf.variable_scope("logit_f"):
            final_features = final_conv_f
            for i, v in enumerate(self.hidden_layers, start=1):
                final_features = tf.layers.dense(
                    inputs=final_features, units=v, name="hidden_{}".format(i),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )

            self.logits_f = tf.layers.dense(
                inputs=final_features, units=self.num_of_class, name="logit_f",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            logits_f = self.batch_normalization(self.logits_f, training=self.is_training)

        with tf.variable_scope("logit_b"):
            final_features = final_conv_b
            for i, v in enumerate(self.hidden_layers, start=1):
                final_features = tf.layers.dense(
                    inputs=final_features, units=v, name="hidden_{}".format(i),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )

            self.logits_b = tf.layers.dense(
                inputs=final_features, units=self.num_of_class, name="logit_b",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            logits_b = self.batch_normalization(self.logits_b, training=self.is_training)

        with tf.variable_scope("logit_final"):
            self.logits = self.alpha * tf.nn.softmax(logits_f)
            self.logits += (1 - self.alpha) * tf.nn.softmax(logits_b)

        with tf.variable_scope("bi_logit"):
            final_features = tf.concat([final_conv_f, final_conv_b], axis=1)
            for i, v in enumerate(self.hidden_layers, start=1):
                final_features = tf.layers.dense(
                    inputs=final_features, units=v, name="hidden_{}".format(i),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )

            self.logits_bi = tf.layers.dense(
                inputs=final_features, units=self.num_of_class_2, name="logit_bi",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            log_likelihood_f = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_f)
            log_likelihood_b = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_b)
            log_likelihood_bi = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_2, logits=self.logits_bi)
            regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = (
                    tf.reduce_mean(log_likelihood_f)
                    + tf.reduce_mean(log_likelihood_b)
                    + tf.reduce_mean(log_likelihood_bi)
            )
            self.loss += tf.reduce_sum(regularizer)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))
            # self.train_op = optimizer.minimize(self.loss)

    def build(self):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholders()
        self._add_word_embeddings_op()

        self._add_logits_op()
        self._add_loss_op()

        self._add_train_op()
        # f = tf.summary.FileWriter("models_summary")
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)
        timer.stop()

    def _loss(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_op] = 1.0
        feed_dict[self.dropout_final_embedding] = 1.0
        feed_dict[self.dropout_dependency] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.is_training] = False

        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            w_batch = data['ws'][start:start + self.batch_size]
            l_batch = data['ls'][start:start + self.batch_size]
            p_batch = data['ps'][start:start + self.batch_size]
            r_batch = data['rs'][start:start + self.batch_size]
            d_batch = data['ds'][start:start + self.batch_size]
            wns_batch = data['wnss'][start:start + self.batch_size]

            char_ids, word_ids = zip(*[zip(*x) for x in w_batch])
            word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_sent_length)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2, max_sent_length=self.max_sent_length)

            labels = l_batch
            labels_2 = [self.LABEL_2_LABEL2_MAP[self.all_labels[y]] for y in labels]
            labels_b = [self.LABEL_2_LABELB_MAP[self.all_labels[y]] for y in labels]

            pos_ids, _ = pad_sequences(p_batch, max_sent_length=self.max_sent_length, pad_tok=0)
            wn_superset_ids, _ = pad_sequences(wns_batch, max_sent_length=self.max_sent_length, pad_tok=0)

            relation_ids, sequence_lengths_re = pad_sequences(r_batch, max_sent_length=self.max_sent_length - 1, pad_tok=0)
            direction_ids, _ = pad_sequences(d_batch, max_sent_length=self.max_sent_length - 1, pad_tok=0)

            start += self.batch_size
            idx += 1
            yield (word_ids, char_ids, labels, labels_b, labels_2, sequence_lengths, sequence_lengths_re, word_lengths,
                   pos_ids, relation_ids, direction_ids, wn_superset_ids)

    def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.train.Saver(max_to_keep=2)
        best_loss = float('inf')
        nepoch_noimp = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch_train = len(self.dataset_train.words) // self.batch_size + 1

            for e in range(epochs):
                words_shuffled, labels_shuffled, poses_shuffled, relations_shuffled, direction_shuffled, wordnet_supersets_shuffled = shuffle(
                    self.dataset_train.words,
                    self.dataset_train.labels,
                    self.dataset_train.poses,
                    self.dataset_train.relations,
                    self.dataset_train.directions,
                    self.dataset_train.wordnet_supersets,
                )

                data = {
                    'ws': words_shuffled,
                    'ls': labels_shuffled,
                    'ps': poses_shuffled,
                    'rs': relations_shuffled,
                    'ds': direction_shuffled,
                    'wnss': wordnet_supersets_shuffled,
                }

                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                    words, chars, labels, labels_b, labels_2, sequence_lengths, sequence_lengths_re, word_lengths, poses, relation, direction, wn_superset = batch
                    feed_dict = {
                        self.word_ids: words,
                        self.wordnet_superset_ids: wn_superset,
                        self.char_ids: chars,
                        self.labels: labels,
                        self.labels_b: labels_b,
                        self.labels_2: labels_2,
                        self.sequence_lens: sequence_lengths,
                        self.relation_lens: sequence_lengths_re,
                        self.word_lengths: word_lengths,
                        self.dropout_op: 0.5,
                        self.dropout_final_embedding: 0.5,
                        self.dropout_dependency: 0.5,
                        self.dropout_cnn: 0.5,
                        self.word_pos_ids: poses,
                        self.relation: relation,
                        self.direction: direction,
                        self.is_training: True,
                        self.is_final: False
                    }

                    _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
                    if idx % 5 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

                Log.log("End epochs {}".format(e + 1))

                # stop by loss
                if early_stopping:
                    num_batch_val = len(self.dataset_validation.words) // self.batch_size + 1
                    total_loss = []

                    data = {
                        'ws': self.dataset_validation.words,
                        'ls': self.dataset_validation.labels,
                        'ps': self.dataset_validation.poses,
                        'rs': self.dataset_validation.relations,
                        'ds': self.dataset_validation.directions,
                        'wnss': self.dataset_validation.wordnet_supersets,
                    }

                    for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                        words, chars, labels, labels_b, labels_2, sequence_lengths, sequence_lengths_re, word_lengths, poses, relation, direction, wn_superset = batch

                        loss = self._loss(sess, feed_dict={
                            self.word_ids: words,
                            self.char_ids: chars,
                            self.wordnet_superset_ids: wn_superset,
                            self.labels: labels,
                            self.labels_b: labels_b,
                            self.labels_2: labels_2,
                            self.sequence_lens: sequence_lengths,
                            self.relation_lens: sequence_lengths_re,
                            self.word_lengths: word_lengths,
                            self.relation: relation,
                            self.direction: direction,
                            self.word_pos_ids: poses,
                            self.is_final: False
                        })

                        total_loss.append(loss)

                    val_loss = np.mean(total_loss)
                    Log.log('Val loss: {}'.format(val_loss))
                    if val_loss < best_loss:
                        saver.save(sess, self.model_name)
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_loss = val_loss
                        nepoch_noimp = 0
                    else:
                        nepoch_noimp += 1
                        Log.log("Number of epochs with no improvement: {}".format(nepoch_noimp))
                        if nepoch_noimp >= patience:
                            Log.log('Best loss: {}'.format(best_loss))
                            break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def load_data(self, train, validation):
        """
        :param dataset.dataset.Dataset train:
        :param dataset.dataset.Dataset validation:
        :return:
        """
        timer = Timer()
        timer.start("Loading data")

        self.dataset_train = train
        self.dataset_validation = validation

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_validation.labels))
        timer.stop()

    def run_train(self, epochs, early_stopping=True, patience=10):
        timer = Timer()
        timer.start("Training model...")
        self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
        timer.stop()

    # test
    def predict(self, test):
        """

        :param dataset.dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log("Testing model over test set")
            # a = tf.train.latest_checkpoint(self.model_name)
            saver.restore(sess, self.model_name)

            y_pred = []
            num_batch = len(test.labels) // self.batch_size + 1

            data = {
                'ws': test.words,
                'ls': test.labels,
                'ps': test.poses,
                'rs': test.relations,
                'ds': test.directions,
                'wnss': test.wordnet_supersets,
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
                words, chars, labels, labels_b, labels_2, sequence_lengths, sequence_lengths_re, word_lengths, poses, relation, direction, wn_superset = batch
                feed_dict = {
                    self.word_ids: words,
                    self.wordnet_superset_ids: wn_superset,
                    self.char_ids: chars,
                    self.word_lengths: word_lengths,
                    self.sequence_lens: sequence_lengths,
                    self.relation_lens: sequence_lengths_re,
                    self.dropout_op: 1.0,
                    self.dropout_final_embedding: 1.0,
                    self.dropout_dependency: 1.0,
                    self.dropout_cnn: 1.0,
                    self.word_pos_ids: poses,
                    self.relation: relation,
                    self.direction: direction,
                    self.is_training: False,
                    self.is_final: False
                }
                logits = sess.run(self.logits, feed_dict=feed_dict)

                for logit in logits:
                    decode_sequence = np.argmax(logit)
                    y_pred.append(decode_sequence)

        return y_pred

    # test
    def infer(self, test):
        """

        :param dataset.dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log("Testing model over test set")
            # a = tf.train.latest_checkpoint(self.model_name)
            saver.restore(sess, self.model_name)

            y_pred = []
            num_batch = len(test.labels) // self.batch_size + 1

            data = {
                'ws': test.words,
                'ls': test.labels,
                'ps': test.poses,
                'rs': test.relations,
                'ds': test.directions,
                'wnss': test.wordnet_supersets,
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
                words, chars, labels, labels_b, labels_2, sequence_lengths, sequence_lengths_re, word_lengths, poses, relation, direction, wn_superset = batch
                feed_dict = {
                    self.word_ids: words,
                    self.wordnet_superset_ids: wn_superset,
                    self.char_ids: chars,
                    self.word_lengths: word_lengths,
                    self.sequence_lens: sequence_lengths,
                    self.relation_lens: sequence_lengths_re,
                    self.dropout_op: 1.0,
                    self.dropout_final_embedding: 1.0,
                    self.dropout_dependency: 1.0,
                    self.dropout_cnn: 1.0,
                    self.word_pos_ids: poses,
                    self.relation: relation,
                    self.direction: direction,
                    self.is_training: False,
                    self.is_final: False
                }
                logits = sess.run(self.logits, feed_dict=feed_dict)

                for logit in logits:
                    y_pred.append(logit)

        return y_pred
