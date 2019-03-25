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
