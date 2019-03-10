import numpy as np
import pickle
import tensorflow as tf
import model.config as config
from model.base_model import BaseModel
import model.util as util
from model.transformer import Transformer


class Model(BaseModel):

    def __init__(self, args, next_element):
        super().__init__(args)
        self.chunk_id, self.words, self.words_len, self.chars, self.chars_len,\
        self.begin_span, self.end_span, self.spans_len,\
        self.cand_entities, self.cand_entities_scores, self.cand_entities_labels,\
        self.cand_entities_len, self.ground_truth, self.ground_truth_len,\
        self.begin_gm, self.end_gm, self.mask_index = next_element

        self.begin_span = tf.cast(self.begin_span, tf.int32)
        self.end_span = tf.cast(self.end_span, tf.int32)
        self.words_len = tf.cast(self.words_len, tf.int32)
        """
        self.words:  tf.int64, shape=[None, None]   # shape = (batch size, max length of sentence in batch)
        self.words_len: tf.int32, shape=[None],     #   shape = (batch size)
        self.chars: tf.int64, shape=[None, None, None], # shape = (batch size, max length of sentence, max length of word)
        self.chars_len: tf.int64, shape=[None, None],   # shape = (batch_size, max_length of sentence)
        self.begin_span: tf.int32, shape=[None, None],  # shape = (batch_size, max number of candidate spans in one of the batch sentences)
        self.end_span: tf.int32, shape=[None, None],
        self.spans_len: tf.int64, shape=[None],     # shape = (batch size)
        self.cand_entities: tf.int64, shape=[None, None, None],  # shape = (batch size, max number of candidate spans, max number of cand entitites)
        self.cand_entities_scores: tf.float32, shape=[None, None, None],
        self.cand_entities_labels: tf.int64, shape=[None, None, None],
        # shape = (batch_size, max number of candidate spans)
        self.cand_entities_len: tf.int64, shape=[None, None],
        self.ground_truth: tf.int64, shape=[None, None],  # shape = (batch_size, max number of candidate spans)
        self.ground_truth_len: tf.int64, shape=[None],    # shape = (batch_size)
        self.begin_gm: tf.int64, shape=[None, None],  # shape = (batch_size, max number of gold mentions)
        self.end_gm = tf.int64, shape=[None, None],
        self.mask_index = tf.int64 shape=[None]  # shape = (batch_size)
        """
        with open(config.base_folder + "data/tfrecords/" + self.args.experiment_name + "/word_char_maps.pickle", 'rb') \
                as handle:
            _, id2word, _, id2char, _, _ = pickle.load(handle)
            self.nwords = len(id2word)
            self.nchars = len(id2char)

        self.loss_mask = self._sequence_mask_v13(self.cand_entities_len, tf.shape(self.cand_entities_scores)[2])

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def init_embeddings(self):
        print("\n!!!! init embeddings !!!!\n")
        # read the numpy file
        embeddings_nparray = np.load(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                                     "/embeddings_array.npy")
        self.sess.run(self.word_embedding_init, feed_dict={self.word_embeddings_placeholder: embeddings_nparray})

        entity_embeddings_nparray = util.load_ent_vecs(self.args)
        self.sess.run(self.entity_embedding_init, feed_dict={self.entity_embeddings_placeholder: entity_embeddings_nparray})

    def add_embeddings_op(self):
        """Defines self.word_embeddings"""
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                    tf.constant(0.0, shape=[self.nwords, 300]),
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=False)

            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [self.nwords, 300])
            self.word_embedding_init = _word_embeddings.assign(self.word_embeddings_placeholder)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.words, name="word_embeddings")
            self.pure_word_embeddings = word_embeddings
            # print("word_embeddings (after lookup) ", word_embeddings)

        with tf.variable_scope("chars"):
            if self.args.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.nchars, self.args.dim_char], trainable=True)
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.chars, name="char_embeddings")

                # char_embeddings: tf.float32, shape=[None, None, None, dim_char],
                # shape = (batch size, max length of sentence, max length of word, dim_char)
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.args.dim_char])
                # (batch*sent_length, characters of word, dim_char)

                char_lengths = tf.reshape(self.chars_len, shape=[s[0] * s[1]])
                # shape = (batch_size*max_length of sentence)

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=char_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[s[0], s[1], 2 * self.args.hidden_size_char])
                # print("output after char lstm ", output)
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)  # concatenate word and char embeddings
                # print("word_embeddings with char after concatenation ", word_embeddings)
                # (batch, words, 300+2*100)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("entities"):
            from preprocessing.util import load_wikiid2nnid
            self.nentities = len(load_wikiid2nnid(extension_name=self.args.entity_extension))
            _entity_embeddings = tf.Variable(
                tf.constant(0.0, shape=[self.nentities, 300]),
                name="_entity_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs)

            self.entity_embeddings_placeholder = tf.placeholder(tf.float32, [self.nentities, 300])
            self.entity_embedding_init = _entity_embeddings.assign(self.entity_embeddings_placeholder)

            self.entity_embeddings = tf.nn.embedding_lookup(_entity_embeddings, self.cand_entities,
                                                       name="entity_embeddings")
            self.pure_entity_embeddings = self.entity_embeddings
            if self.args.ent_vecs_regularization.startswith("l2"):  # 'l2' or 'l2dropout'
                self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, dim=3)
                # not necessary since i do normalization in the entity embed creation as well, just for safety
            if self.args.ent_vecs_regularization == "dropout" or \
                            self.args.ent_vecs_regularization == "l2dropout":
                self.entity_embeddings = tf.nn.dropout(self.entity_embeddings, self.dropout)
            # print("entity_embeddings = ", self.entity_embeddings)

    def add_context_tr_emb_op(self):
        hparams = {"num_units": 300, "dropout": 1 - self.dropout, "is_training": True,
                   "num_multi_head": 1, "num_heads": 4, "max_seq_len": 3000}
        with tf.variable_scope("context-bi-transformer"):
            transformer = Transformer(hparams)
            output = transformer.encoder(self.word_embeddings, self.words_len)
            self.context_emb = output

    def add_span_emb_op(self):
        mention_emb_list = []
        # span embedding based on boundaries (start, end) and head mechanism. but do that on top of contextual bilistm
        # output or on top of original word+char embeddings. this flag determines that. The parer reports results when
        # using the contextual lstm emb as it achieves better score. Used for ablation studies.
        boundaries_input_vecs = self.word_embeddings if self.args.span_boundaries_from_wordemb else self.context_emb

        # the span embedding is modeled by g^m = [x_q; x_r; \hat(x)^m]  (formula (2) of paper)
        # "boundaries" mean use x_q and x_r.   "head" means use also the head mechanism \hat(x)^m (formula (3))
        if self.args.span_emb.find("boundaries") != -1:
            # shape (batch, num_of_cand_spans, emb)
            mention_start_emb = tf.gather_nd(boundaries_input_vecs, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span)[0]), 1), [1, tf.shape(self.begin_span)[1]]),
                 self.begin_span], 2))  # extracts the x_q embedding for each candidate span
            # the tile command creates a 2d tensor with the batch information. first lines contains only zeros, second
            # line ones etc...  because the begin_span tensor has the information which word inside this sentence is the
            # beginning of the candidate span.
            mention_emb_list.append(mention_start_emb)

            mention_end_emb = tf.gather_nd(boundaries_input_vecs, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span)[0]), 1), [1, tf.shape(self.begin_span)[1]]),
                 tf.nn.relu(self.end_span-1)], 2))   # -1 because the end of span in exclusive  [start, end)
            # relu so that the 0 doesn't become -1
            # of course no valid candidate span end index is zero since [0,0) is empty
            mention_emb_list.append(mention_end_emb)
            # print("mention_start_emb = ", mention_start_emb)
            # print("mention_end_emb = ", mention_end_emb)

        mention_width = self.end_span - self.begin_span  # [batch, num_mentions]     the width of each candidate span
        self.span_emb = tf.concat(mention_emb_list, 2) # [batch, num_mentions, emb i.e. 1700] formula (2) concatenation
        # print("span_emb = ", self.span_emb)

    def add_loss_op(self):
        cand_entities_labels = tf.cast(self.cand_entities_labels, tf.float32)
        loss1 = cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores)
        loss2 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores)
        self.loss = loss1 + loss2
        if self.args.nn_components.find("global") != -1 and not self.args.global_one_loss:
            loss3 = cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores_before_global)
            loss4 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores_before_global)
            self.loss = loss1 + loss2 + loss3 + loss4
        # print("loss_mask = ", loss_mask)
        self.loss = self.loss_mask * self.loss
        self.loss = tf.reduce_sum(self.loss)
        # for tensorboard
        # tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_embeddings_op()
        self.add_context_tr_emb_op()
        self.add_span_emb_op()
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

    def _sequence_mask_v13(self, mytensor, max_width):
        """mytensor is a 2d tensor"""
        temp_mask = tf.sequence_mask(mytensor, max_width, dtype=tf.float32)
        return temp_mask
