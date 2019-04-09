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
        with tf.variable_scope("input_fn"):
            """Define placeholders = entries to computational graph"""
            self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.chunk_id = tf.placeholder(tf.string, [None], name="chunk_id")
            self.words = tf.placeholder(tf.int64, [None, None], name="words")
            self.words_len = tf.placeholder(tf.int64, [None], name="words_len")
            self.chars = tf.placeholder(tf.int64, [None, None, None], name="chars")
            self.chars_len = tf.placeholder(tf.int64, [None, None], name="chars_len")
            self.begin_span = tf.placeholder(tf.int64, [None, None], name="begin_span")
            self.end_span = tf.placeholder(tf.int64, [None, None], name="end_span")
            self.spans_len = tf.placeholder(tf.int64, [None], name="spans_len")
            self.cand_entities = tf.placeholder(tf.int64, [None, None, None], name="cand_entities")
            self.cand_entities_scores = tf.placeholder(tf.float32, [None, None, None], name="cand_entities_scores")
            self.cand_entities_labels = tf.placeholder(tf.int64, [None, None, None], name="cand_entities_labels")
            self.cand_entities_len = tf.placeholder(tf.int64, [None, None], name="cand_entities_len")
            self.ground_truth = tf.placeholder(tf.int64, [None, None], name="ground_truth")
            self.ground_truth_len = tf.placeholder(tf.int64, [None], name="ground_truth_len")
            self.begin_gm = tf.placeholder(tf.int64, [None, None], name="begin_gm")
            self.end_gm = tf.placeholder(tf.int64, [None, None], name="end_gm")
            self.mask_index = tf.placeholder(tf.int64, [None], name="mask_index")
            self.entities = tf.placeholder(tf.string, [None, None], name="entities")
            self.entities_only = tf.placeholder(tf.string, [None, None], name="entities_only")
            self.cand_local_scores = tf.placeholder(tf.float32, [None, None, None], name="local_scores")
            self.mask_ent_id = tf.placeholder(tf.string, [None], name="mask_ent_id")

            # slice candidate entities of mask index
            # shape = [batch_size, #cand_entitites]
            self.mask_cand_entities = self.extract_axis_1(self.cand_entities, self.mask_index)
            self.mask_cand_entities_labels = self.extract_axis_1(self.cand_entities_labels, self.mask_index)
            self.mask_cand_entities_labels = tf.cast(self.mask_cand_entities_labels, tf.float32)
            self.mask_cand_local_scores = self.extract_axis_1(self.cand_local_scores, self.mask_index)
            # shape = [batch_size]
            self.mask_cand_entities_len = self.extract_axis_1(self.cand_entities_len, self.mask_index)
            self.mask_begin_span = self.extract_axis_1(self.begin_span, self.mask_index)
            self.mask_end_span = self.extract_axis_1(self.end_span, self.mask_index)

            # split entities for ground truth
            # shape = [batch_size, word_length, 1/3]
            self.mask_entities = tf.string_split(tf.reshape(self.entities, [-1]), '_').values
            self.mask_entities = tf.reshape(self.mask_entities, [tf.shape(self.entities)[0], tf.shape(self.entities)[1], -1])
            self.mask_entities = tf.string_to_number(self.mask_entities, tf.int64)

            self.mask_entities_only = tf.string_split(tf.reshape(self.entities_only, [-1]), '_').values
            self.mask_entities_only = tf.reshape(self.mask_entities_only, [tf.shape(self.entities_only)[0], tf.shape(self.entities_only)[1], -1])
            self.mask_entities_only = tf.string_to_number(self.mask_entities_only, tf.int64)

            # loss mask
            self.loss_mask = tf.sequence_mask(self.mask_cand_entities_len, tf.shape(self.mask_cand_entities)[1], dtype=tf.float32)

        with tf.variable_scope("next_example"):
            self.next_data = next_element

    def extract_axis_2(self, data, ind):
        data_t = tf.transpose(data, [0, 2, 1])
        res = self.extract_axis_1(data_t, ind)
        return res

    def extract_axis_1(self, data, ind):
        batch_range = tf.range(tf.shape(data, out_type=tf.int64)[0], dtype=tf.int64)
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def init_embeddings(self):
        print("\n!!!! init embeddings !!!!\n")
        # read the numpy file
        embeddings_nparray = np.load(config.base_folder + "data/tfrecords/" + self.args.experiment_name + "/embeddings_array.npy")
        entity_embeddings_nparray = util.load_ent_vecs(self.args)

        with tf.variable_scope("init_embeddings"):
            self.sess.run(self.word_embedding_init, feed_dict={self.word_embeddings_placeholder: embeddings_nparray})
            self.sess.run(self.entity_embedding_init, feed_dict={self.entity_embeddings_placeholder: entity_embeddings_nparray})

    def add_embeddings_op(self):
        with open(config.base_folder + "data/tfrecords/" + self.args.experiment_name + "/word_char_maps.pickle", 'rb') as handle:
            _, id2word, _, id2char, _, _ = pickle.load(handle)
            nwords = len(id2word)
            nentities = len(load_wikiid2nnid(extension_name=self.args.entity_extension))
            self.entity_embeddings_placeholder = tf.placeholder(tf.float32, [nentities, 300])
            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [nwords, 300])

        """Defines self.cand/entity_embeddings"""
        with tf.variable_scope("entity_embeddings"):
            # with tf.device("/cpu:0"):
            _entity_embeddings = tf.Variable(
                tf.constant(0.0, shape=[nentities, 300]),
                name="_entity_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs)
            _entity_default_embeddings = tf.Variable(
                tf.constant(0.0, shape=[1, 300]),
                name="_entity_default_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs
            )
            self.entity_embedding_init = _entity_embeddings.assign(self.entity_embeddings_placeholder)
            _new_entity_embeddings = tf.concat([_entity_embeddings, _entity_default_embeddings], axis=0)

            # for classification
            self.cand_entity_embeddings = tf.nn.embedding_lookup(_new_entity_embeddings, self.mask_cand_entities, name="cand_entity_embeddings")
            # input entity
            entity_embeddings = tf.nn.embedding_lookup(_new_entity_embeddings, self.mask_entities, name="entity_embeddings")
            entity_embeddings = tf.reduce_mean(entity_embeddings, axis=-2)
            # entity sequence
            entity_only_embeddings = tf.nn.embedding_lookup(_new_entity_embeddings, self.mask_entities_only, name="entity_only_embeddings")
            self.entity_only_embeddings = tf.reduce_mean(entity_only_embeddings, axis=-2)
            # global entity
            global_entity_embeddings = tf.reduce_sum(self.entity_only_embeddings, axis=-2)
            self.global_entity_embeddings = tf.nn.l2_normalize(global_entity_embeddings, dim=-1)

        """Defines self.word_embeddings"""
        with tf.variable_scope("word_embeddings"):
            # with tf.device("/cpu:0"):
            _word_embeddings = tf.Variable(
                tf.constant(0.0, shape=[nwords, 300]),
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=False)
            self.word_embedding_init = _word_embeddings.assign(self.word_embeddings_placeholder)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.words, name="word_embeddings")
            self.word_embeddings = word_embeddings + entity_embeddings

    def add_context_tr_emb_op(self):
        hparams = {"num_units": 300, "dropout": 1 - self.dropout, "is_training": True,
                   "num_multi_head": 1, "num_heads": 3, "max_seq_len": 10000}
        with tf.variable_scope("context-bi-transformer", reuse=tf.AUTO_REUSE):
            transformer = Transformer(hparams)
            output = transformer.encoder(self.word_embeddings, self.words_len)

        with tf.variable_scope("mask_span_embedding", reuse=tf.AUTO_REUSE):
            mention_emb_list = []
            # span embedding based on boundaries (start, end) and head mechanism.
            boundaries_input_vecs = output
            # the span embedding is modeled by g^m = [x_q; x_r]
            if self.args.span_emb.find("boundaries") != -1:
                # shape = [batch, emb]
                mention_start_emb = self.extract_axis_1(boundaries_input_vecs, self.mask_begin_span)
                mention_emb_list.append(mention_start_emb)
                mention_end_emb = self.extract_axis_1(boundaries_input_vecs, self.mask_end_span - 1)
                mention_emb_list.append(mention_end_emb)
            # shape = [batch_size, 300]
            self.span_emb = tf.layers.dense(tf.concat(mention_emb_list, -1), 300)

    def slice_k(self, mask_begin, embeddings, k):
        k = tf.minimum(tf.shape(embeddings)[1], k)
        # [batch]
        k_begin = tf.nn.relu(tf.cast(mask_begin, tf.int32) - k)
        # [2k] + [batch, 1] -> [batch, 2k]
        k_indices = tf.minimum(tf.shape(embeddings)[1] - 1, tf.range(2 * k) + tf.expand_dims(k_begin, 1))
        # [batch, 2k]
        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(k_indices)[0]), 1), [1, tf.shape(k_indices)[1]])
        # [batch, 2k, 2]
        window_indices = tf.stack([batch_index, k_indices], 2)
        window_word_embeddings = tf.gather_nd(embeddings, window_indices)
        return window_word_embeddings, tf.cast(k_begin, tf.int64)

    def add_context_tr_window(self):
        hparams = {"num_units": 300, "dropout": 1 - self.dropout, "is_training": True,
                   "num_multi_head": 1, "num_heads": 3, "max_seq_len": 10000}
        with tf.variable_scope("context-bi-transformer", reuse=tf.AUTO_REUSE):
            transformer = Transformer(hparams)
            window_word_embeddings, k_begin = self.slice_k(self.mask_begin_span, self.word_embeddings, 50)
            output = transformer.encoder(window_word_embeddings, tf.minimum(self.words_len, k_begin + tf.cast(2 * 50, tf.int64)) - k_begin)

        with tf.variable_scope("mask_span_embedding", reuse=tf.AUTO_REUSE):
            mention_begin = self.mask_begin_span - k_begin
            mention_end = self.mask_end_span - k_begin
            mention_start_emb = self.extract_axis_1(output, mention_begin)
            mention_end_emb = self.extract_axis_1(output, mention_end - 1)
            self.window_span_emb = tf.layers.dense(tf.concat([mention_start_emb, mention_end_emb], -1), 300)

    def add_entity_tr_window(self):
        hparams = {"num_units": 300, "dropout": 1 - self.dropout, "is_training": True,
                   "num_multi_head": 1, "num_heads": 3, "max_seq_len": 20}
        with tf.variable_scope("entity-bi-transformer"):
            transformer = Transformer(hparams)
            window_entity_embeddings, k_begin = self.slice_k(self.mask_index, self.entity_only_embeddings, 3)
            window_entity_embeddings = tf.concat([window_entity_embeddings, tf.expand_dims(self.span_emb, 1)], axis=0)
            output = transformer.encoder(window_entity_embeddings, tf.minimum(self.spans_len, k_begin + tf.cast(2 * 3, tf.int64)) - k_begin)
            self.window_entity_emb = self.extract_axis_1(output, tf.cast(tf.shape(output)[1] - 1 + tf.zeros([tf.shape(output)[0]]), tf.int64))
            # window_entity_emb = tf.reduce_sum(output, axis=-2)
            # self.window_entity_emb = tf.nn.l2_normalize(window_entity_emb, dim=-1)

    def add_final_score_op(self):
        with tf.variable_scope("final_score"):
            # context-aware global scores => [batch_size, #cands, 300] * [batch_size, 1, 300] = [batch_size, #cands, 1]
            global_context_scores = tf.matmul(self.cand_entity_embeddings, tf.expand_dims(self.span_emb, 1), transpose_b=True)
            # global voting sores => [batch_size, #cands, 1]
            global_voting_scores = tf.matmul(self.cand_entity_embeddings, tf.expand_dims(self.global_entity_embeddings, 1), transpose_b=True)
            global_window_context_scores = tf.matmul(self.cand_entity_embeddings, tf.expand_dims(self.window_span_emb, 1), transpose_b=True)
            global_window_voting_scores = tf.matmul(self.cand_entity_embeddings, tf.expand_dims(self.window_entity_emb, 1), transpose_b=True)
            final_scores = tf.layers.dense(tf.concat([global_context_scores, global_voting_scores,
                                                      global_window_context_scores, global_window_voting_scores], axis=-1), 1)
            self.final_scores = tf.squeeze(final_scores, axis=-1)

    def add_loss_op(self):
        with tf.variable_scope("loss"):
            loss1 = self.mask_cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores)
            loss2 = (1 - self.mask_cand_entities_labels) * tf.nn.relu(self.final_scores)
            loss = loss1 + loss2
            loss = self.loss_mask * loss
            self.loss = tf.reduce_sum(loss)

    def build(self):
        self.add_embeddings_op()
        self.add_context_tr_emb_op()
        self.add_context_tr_window()
        self.add_entity_tr_window()
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
