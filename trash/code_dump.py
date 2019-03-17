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
