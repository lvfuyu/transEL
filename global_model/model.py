import numpy as np
import pickle
import tensorflow as tf
import model.config as config
from model.base_model import BaseModel
import model.util as util
from model.transformer import Transformer
from preprocessing.util import load_wikiid2nnid


class Model(BaseModel):

    def __init__(self, args, next_element):
        super().__init__(args)
        """
        self.words:  tf.int64, shape=[None, None]  # shape = (batch size, max length of sentence in batch)
        self.words_len: tf.int32, shape=[None],  # shape = (batch size)
        self.chars: tf.int64, shape=[None, None, None],  # shape = (batch size, max length of sentence, max length of word)
        self.chars_len: tf.int64, shape=[None, None],  # shape = (batch_size, max_length of sentence)
        self.begin_span: tf.int32, shape=[None, None],  # shape = (batch_size, max number of candidate spans in one of the batch sentences)
        self.end_span: tf.int32, shape=[None, None],
        self.spans_len: tf.int64, shape=[None],  # shape = (batch size)
        self.cand_entities: tf.int64, shape=[None, None, None],  # shape = (batch size, max number of candidate spans, max number of cand entitites)
        self.cand_entities_scores: tf.float32, shape=[None, None, None],
        self.cand_entities_labels: tf.int64, shape=[None, None, None], 
        self.cand_entities_len: tf.int64, shape=[None, None],  # shape = (batch_size, max number of candidate spans)
        self.ground_truth: tf.int64, shape=[None, None],  # shape = (batch_size, max number of candidate spans)
        self.ground_truth_len: tf.int64, shape=[None],  # shape = (batch_size)
        self.begin_gm: tf.int64, shape=[None, None],  # shape = (batch_size, max number of gold mentions)
        self.end_gm = tf.int64, shape=[None, None],
        self.mask_index = tf.int64 shape=[None]  # shape = (batch_size)
        """
        with tf.variable_scope("input_fn"):
            self.chunk_id, self.words, self.words_len, self.chars, self.chars_len,\
            self.begin_span, self.end_span, self.spans_len,\
            self.cand_entities, self.cand_entities_scores, self.cand_entities_labels,\
            self.cand_entities_len, self.ground_truth, self.ground_truth_len,\
            self.begin_gm, self.end_gm, self.mask_index, self.entities = next_element

            # type preprocess
            self.begin_span = tf.cast(self.begin_span, tf.int32)
            self.end_span = tf.cast(self.end_span, tf.int32)
            self.words_len = tf.cast(self.words_len, tf.int32)
            self.cand_entities_labels = tf.cast(self.cand_entities_labels, tf.float32)
            self.mask_index = tf.cast(self.mask_index, tf.int32)

            # slice cand entities of mask index
            # shape = [batch_size, max number of cand entitites]
            self.cand_entities = self.extract_axis_1(self.cand_entities, self.mask_index)
            self.cand_entities_labels = self.extract_axis_1(self.cand_entities_labels, self.mask_index)
            # shape = [batch_size]
            self.cand_entities_len = self.extract_axis_1(self.cand_entities_len, self.mask_index)
            self.begin_span = self.extract_axis_1(self.begin_span, self.mask_index)
            self.end_span = self.extract_axis_1(self.end_span, self.mask_index)
            # loss mask
            self.loss_mask = tf.sequence_mask(self.cand_entities_len, tf.shape(self.cand_entities)[1], dtype=tf.float32)

        with open(config.base_folder + "data/tfrecords/" + self.args.experiment_name + "/word_char_maps.pickle", 'rb') as handle:
            _, id2word, _, id2char, _, _ = pickle.load(handle)
            self.nwords = len(id2word)
            self.nchars = len(id2char)
            self.nentities = len(load_wikiid2nnid(extension_name=self.args.entity_extension))

    def extract_axis_1(self, data, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        batch_range = tf.range(tf.shape(data)[0], dtype=tf.int32)
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def init_embeddings(self):
        print("\n!!!! init embeddings !!!!\n")
        # read the numpy file
        embeddings_nparray = np.load(config.base_folder + "data/tfrecords/" +
                                     self.args.experiment_name + "/embeddings_array.npy")
        entity_embeddings_nparray = util.load_ent_vecs(self.args)

        with tf.variable_scope("init_embeddings"):
            self.sess.run(self.word_embedding_init,
                          feed_dict={self.word_embeddings_placeholder: embeddings_nparray})

            self.sess.run(self.entity_embedding_init,
                          feed_dict={self.entity_embeddings_placeholder: entity_embeddings_nparray})

    def add_embeddings_op(self):
        """Defines self.cand/entity_embeddings"""
        with tf.variable_scope("entity_embeddings"):
            _entity_embeddings = tf.Variable(
                tf.constant(0.0, shape=[self.nentities, 300]),
                name="_entity_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs)
            _entity_default_embeddings = tf.Variable(
                tf.constant(0.0, shape=[1, 300]),
                name="_entity_default_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs
            )
            self.entity_embeddings_placeholder = tf.placeholder(tf.float32, [self.nentities, 300])
            self.entity_embedding_init = _entity_embeddings.assign(self.entity_embeddings_placeholder)
            _new_entity_embeddings = tf.concat([_entity_embeddings, _entity_default_embeddings], axis=0)

            # for classification
            self.cand_entity_embeddings = tf.nn.embedding_lookup(_new_entity_embeddings, self.cand_entities,
                                                                 name="cand_entity_embeddings")
            # input entity
            entity_embeddings = tf.nn.embedding_lookup(_new_entity_embeddings, self.entities,
                                                       name="entity_embeddings")

        """Defines self.word_embeddings"""
        with tf.variable_scope("word_embeddings"):
            _word_embeddings = tf.Variable(
                tf.constant(0.0, shape=[self.nwords, 300]),
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=False)
            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [self.nwords, 300])
            self.word_embedding_init = _word_embeddings.assign(self.word_embeddings_placeholder)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.words, name="word_embeddings")
            self.word_embeddings = word_embeddings + entity_embeddings

    def add_context_tr_emb_op(self):
        hparams = {"num_units": 300, "dropout": 1 - self.dropout, "is_training": True,
                   "num_multi_head": 1, "num_heads": 4, "max_seq_len": 3000}
        with tf.variable_scope("context-bi-transformer"):
            transformer = Transformer(hparams)
            output = transformer.encoder(self.word_embeddings, self.words_len)
            self.context_emb = output

    def add_span_emb_op(self):
        with tf.variable_scope("mask_span_embedding"):
            mention_emb_list = []
            # span embedding based on boundaries (start, end) and head mechanism.
            boundaries_input_vecs = self.context_emb
            # the span embedding is modeled by g^m = [x_q; x_r]
            if self.args.span_emb.find("boundaries") != -1:
                # shape = [batch, emb]
                mention_start_emb = self.extract_axis_1(boundaries_input_vecs, self.begin_span)
                mention_emb_list.append(mention_start_emb)
                mention_end_emb = self.extract_axis_1(boundaries_input_vecs, self.end_span)
                mention_emb_list.append(mention_end_emb)
            # shape = [batch_size, 300]
            self.span_emb = util.projection(tf.concat(mention_emb_list, -1), 300)

    def add_final_score_op(self):
        with tf.variable_scope("final_score"):
            # [batch_size, 1, 300] * [batch_size, #cands, 300]
            scores = tf.matmul(tf.expand_dims(self.span_emb, 1), self.cand_entity_embeddings, transpose_b=True)
            # [batch_size, #cands]
            self.final_scores = tf.squeeze(scores, axis=1)

    def add_loss_op(self):
        with tf.variable_scope("loss"):
            loss1 = self.cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores)
            loss2 = (1 - self.cand_entities_labels) * tf.nn.relu(self.final_scores)
            loss = loss1 + loss2
            loss = self.loss_mask * loss
            self.loss = tf.reduce_sum(loss)

    def build(self):
        self.add_placeholders()
        self.add_embeddings_op()
        self.add_context_tr_emb_op()
        self.add_span_emb_op()
        self.add_final_score_op()
        if self.args.running_mode.startswith("train"):
            self.add_loss_op()
            # Generic functions that add training op
            self.add_train_op(self.args.lr_method, self.lr, self.loss, self.args.clip)
            self.merged_summary_op = tf.summary.merge_all()

        if self.args.running_mode == "train_continue":
            self.restore_session("latest")
        elif self.args.running_mode == "train":
            # now self.sess is defined and vars are init
            self.initialize_session()
            self.init_embeddings()
