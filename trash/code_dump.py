# if self.args.ent_vecs_regularization == "dropout" or self.args.ent_vecs_regularization == "l2dropout":
#     self.entity_embeddings = tf.nn.dropout(entity_embeddings, self.dropout)
#     self.cand_entity_embeddings = tf.nn.dropout(cand_entity_embeddings, self.dropout)
# self.word_embeddings if self.args.span_boundaries_from_wordemb else
# mention_width = self.end_span - self.begin_span  # [batch, num_mentions]     the width of each candidate span
# print("span_emb = ", self.span_emb)
# [batch, num_mentions, emb i.e. 1700] formula (2) concatenation
# print("mention_start_emb = ", mention_start_emb)
# print("mention_end_emb = ", mention_end_emb)
# tf_writers["ed_pr"] = tf.summary.FileWriter(args.summaries_folder + 'ed_pr/')
# tf_writers["ed_re"] = tf.summary.FileWriter(args.summaries_folder + 'ed_re/')
# tf_writers["ed_f1"] = tf.summary.FileWriter(args.summaries_folder + 'ed_f1/')
# def count_records_of_one_epoch(trainfiles):
#     filename_queue = tf.train.string_input_producer(trainfiles, num_epochs=1)
#     reader = tf.TFRecordReader()
#     key, serialized_example = reader.read(filename_queue)
#     with tf.Session() as sess:
#         sess.run(
#             tf.variables_initializer(
#                 tf.global_variables() + tf.local_variables()
#             )
#         )
#
#         # Start queue runners
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         counter = 0
#         try:
#             while not coord.should_stop():
#                 fetch_vals = sess.run((key))
#                 # print(fetch_vals)
#                 counter += 1
#         except tf.errors.OutOfRangeError:
#             pass
#         except KeyboardInterrupt:
#             print("Training stopped by Ctrl+C.")
#         finally:
#             coord.request_stop()
#         coord.join(threads)
#     print("number of tfrecords in trainfiles = ", counter)
#     return counter
# if __name__ == "__main__":
#     count_records_of_one_epoch(["/users/guotonglei/entity_linking/end2end_neural_el/data/tfrecords/"
#                                 "corefmerge/"
#                                 "gmonly/wikipedia"])
# type preprocess
# self.begin_span = tf.cast(self.begin_span, tf.int32)
# self.end_span = tf.cast(self.end_span, tf.int32)
# self.words_len = tf.cast(self.words_len, tf.int32)
# self.cand_entities_labels = tf.cast(self.cand_entities_labels, tf.float32)
# self.mask_index = tf.cast(self.mask_index, tf.int32)
# print(next_element)
# for training based on training steps
# for _ in range(args.steps_before_evaluation):
# print(tf.global_variables())
# retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
#               model.begin_span, model.end_span, model.spans_len,
#               model.begin_gm, model.end_gm,
#               model.ground_truth, model.ground_truth_len,
#               model.words_len, model.chunk_id]
# result_l = model.sess.run(
#     retrieve_l, feed_dict={model.input_handle_ph: dataset_handle, model.dropout: 1})

# def parse_sequence_example_test(serialized):
#     sequence_features = {
#             "words": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#             "chars": tf.VarLenFeature(tf.int64),
#             "chars_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#             "entities": tf.FixedLenSequenceFeature([], dtype=tf.string),  # entity ids
#             "begin_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # mention positions
#             "end_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#             "cand_entities": tf.VarLenFeature(tf.int64),
#             "cand_entities_scores": tf.VarLenFeature(tf.float32),
#             "cand_entities_labels": tf.VarLenFeature(tf.int64),
#             "cand_entities_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#             "ground_truth": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#             "begin_gm": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#             "end_gm": tf.FixedLenSequenceFeature([], dtype=tf.int64)
#     }
#
#     context, sequence = tf.parse_single_sequence_example(
#         serialized,
#         context_features={
#             "chunk_id": tf.FixedLenFeature([], dtype=tf.string),
#             "words_len": tf.FixedLenFeature([], dtype=tf.int64),
#             "spans_len": tf.FixedLenFeature([], dtype=tf.int64),
#             "ground_truth_len": tf.FixedLenFeature([], dtype=tf.int64),
#             "mask_index": tf.FixedLenFeature([], dtype=tf.int64),
#         },
#         sequence_features=sequence_features)
#
#     return context["chunk_id"], sequence["words"], context["words_len"],\
#            tf.sparse_tensor_to_dense(sequence["chars"]), sequence["chars_len"],\
#            sequence["begin_span"], sequence["end_span"], context["spans_len"],\
#            tf.sparse_tensor_to_dense(sequence["cand_entities"]),\
#            tf.sparse_tensor_to_dense(sequence["cand_entities_scores"]),\
#            tf.sparse_tensor_to_dense(sequence["cand_entities_labels"]),\
#            sequence["cand_entities_len"], sequence["ground_truth"], context["ground_truth_len"],\
#            sequence["begin_gm"], sequence["end_gm"],\
#            context["mask_index"], sequence["entities"]
# print(len(entities[0]))

#--no_pre_training \
# --training_name=group_global/global_trans_1_4_model_v$v \
# --hardcoded_thr=-100
# in order to have a vector. if i put [1] it will probably be a matrix with just one column
# summary = sess.run(model.merged_summary_op)
            # tf_writers["train"].add_summary(summary, args.eval_cnt)
# comparison_ed_score = (ed_scores[1] + ed_scores[4]) / 2   # aida_dev + acquaint
            # comparison_score = ed_scores[1]  # aida_dev
# if not args.hardcoded_thr and len(val_datasets) == 1 and abs(micro_results[val_datasets[0]] - val_f1) > 0.1:
    #     print("ASSERTION ERROR: optimal threshold f1 calculalation differs from normal"
    #           "f1 calculation!!!!", val_f1, "  and ", micro_results[val_datasets[0]])
# for test_handle, test_name, test_it in zip(handles, names, iterators):
# name is the name of the dataset e.g. aida_test.txt, aquaint.txt
    # Run one pass over the validation dataset.
# print(entities)
    # print(span_len)
    # print(begin_span)
    # print(end_span)
    # print(np.array(begin_span))
    # print(np.array(end_span))
    # print(result_l[0][0])
    # print(next_data[1])
# print(k)
# print(mask_entities)
# print(np.array(local_entities[0][begin_span[0][i]]))
# if args.use_local and args.eval_cnt > 12:
#     result_l[0][0][i] = 0.5*result_l[0][0][i] + 0.5*local_scores[0]
# print(begin_span[i])
# print(end_span[i])
# just for convenience so i can access it from everywhere
# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
# local_scores
# , model.local_scores
# local_entity_emb = tf.nn.l2_normalize(self.local_entity_embeddings, dim=-1)
# local_scores = tf.matmul(tf.expand_dims(local_entity_emb, 1), self.cand_entity_embeddings, transpose_b=True)
# self.local_scores = tf.squeeze(local_scores, axis=1)
# if self.args.use_local:
#     self.final_scores = tf.expand_dims(self.final_scores, axis=2)
#     self.local_scores = tf.expand_dims(self.local_scores, axis=2)
#     self.final_scores = tf.concat([self.final_scores, self.local_scores], axis=-1)
#     self.final_scores = tf.layers.dense(self.final_scores, 1)
#     self.final_scores = tf.squeeze(self.final_scores, -1)
# pred_entity_emb = tf.layers.dense(pred_entity_emb, 300, tf.nn.relu)
# pred_entity_emb = tf.nn.dropout(pred_entity_emb, keep_prob=self.dropout)


# def validation(model, dataset_handle):
#     next_data = model.sess.run([model.next_data], feed_dict={model.input_handle_ph: dataset_handle})
#     next_data = next_data[0]
#     result_l = [next_data[9], next_data[11], next_data[8], next_data[5], next_data[6], next_data[7],
#                 next_data[14], next_data[15], next_data[12], next_data[13], next_data[2], next_data[0]]
#
#     # batch_size = 1
#     begin_span = np.array(next_data[5])
#     end_span = np.array(next_data[6])
#     span_len = next_data[7][0]
#     entities = next_data[17]
#     local_entities = np.copy(entities)
#
#     for k in range(50):
#         entities_tmp = np.copy(entities)
#         flag = True
#         for i in range(span_len):
#             mask_index = np.array([i])
#             mask_entities = np.copy(entities_tmp)
#
#             pred_scores, cand_entities_len, cand_entities = \
#                 model.sess.run([model.final_scores, model.mask_cand_entities_len, model.mask_cand_entities],
#                                feed_dict={model.dropout: 1,
#                                           model.chunk_id: next_data[0],
#                                           model.words: next_data[1],
#                                           model.words_len: next_data[2],
#                                           model.chars: next_data[3],
#                                           model.chars_len: next_data[4],
#                                           model.begin_span: next_data[5],
#                                           model.end_span: next_data[6],
#                                           model.spans_len: next_data[7],
#                                           model.cand_entities: next_data[8],
#                                           model.cand_entities_scores: next_data[9],
#                                           model.cand_entities_labels: next_data[10],
#                                           model.cand_entities_len: next_data[11],
#                                           model.ground_truth: next_data[12],
#                                           model.ground_truth_len: next_data[13],
#                                           model.begin_gm: next_data[14],
#                                           model.end_gm: next_data[15],
#                                           model.mask_index: mask_index,
#                                           model.entities: mask_entities,
#                                           model.local_entities: np.array([local_entities[0][begin_span[0][i]]])})
#             result_l[0][0][i] = pred_scores[0]
#
#             max_score = float('-inf')
#             top_1_entity = -1
#             for j in range(cand_entities_len[0]):
#                 if max_score < pred_scores[0][j]:
#                     top_1_entity = cand_entities[0][j]
#                     max_score = pred_scores[0][j]
#
#             for j in range(begin_span[0][i], end_span[0][i]):
#                 if str(entities[0][j]) != str(top_1_entity):
#                     entities[0][j] = str(top_1_entity)
#                     flag = False
#
#         if k == 49:
#             print(next_data[0], "inference_iter:", k)
#         if flag:
#             break
#
#         if k == 0:
#             default_mask = "502661"
#             for i in range(len(entities[0])):
#                 if entities[0][i] == b'502661_502661_502661':
#                     entities[0][i] = default_mask
#
#     return result_l

# context["mask_ent_id"]]

# "502661_502661_502661"

# self.local_entities = tf.placeholder(tf.string, [None], name="local_entities")
"""
self.words:  tf.int64, shape=[None, None]  # shape = (batch size, max length of sentence in batch)
self.words_len: tf.int64, shape=[None],  # shape = (batch size)
self.chars: tf.int64, shape=[None, None, None],  # shape = (batch size, max length of sentence, max length of word)
self.chars_len: tf.int64, shape=[None, None],  # shape = (batch_size, max_length of sentence)
self.begin_span: tf.int64, shape=[None, None],  # shape = (batch_size, max number of candidate spans in one of the batch sentences)
self.end_span: tf.int64, shape=[None, None],
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

# shape = [batch_size, 3]
# self.mask_local_entities = tf.string_split(tf.reshape(self.local_entities, [-1]), "_").values
# self.mask_local_entities = tf.reshape(self.mask_local_entities, [tf.shape(self.words)[0], -1])
# self.mask_local_entities = self.extract_axis_1(self.mask_local_entities, tf.zeros([tf.shape(self.words)[0]], dtype=tf.int64))
# self.mask_local_entities = tf.reshape(self.mask_local_entities, [tf.shape(self.words)[0], 1])
# self.mask_local_entities = tf.string_to_number(self.mask_local_entities, tf.int64)

# self.mask_entities = self.extract_axis_2(self.mask_entities, tf.zeros([tf.shape(self.words)[0]], dtype=tf.int64))
# self.mask_entities = tf.reshape(self.mask_entities, [tf.shape(self.words)[0], tf.shape(self.words)[1], 1])
# split entities for ground truth and local predictions

# local prediction entities
# tf.nn.embedding_lookup(_new_entity_embeddings, self.mask_local_entities, name="local_entity_embeddings")
# self.local_entity_embeddings =

# pred_entity_emb = tf.nn.l2_normalize(self.span_emb, dim=-1)

# model.cand_local_scores: np.array([local_entities[0][begin_span[0][i]]])})

# local_entities = np.copy(entities)
