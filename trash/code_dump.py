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
