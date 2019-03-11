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
