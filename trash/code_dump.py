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
