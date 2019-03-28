import tensorflow as tf


def parse_sequence_example(serialized):
    sequence_features = {
            "words": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chars": tf.VarLenFeature(tf.int64),
            "chars_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "entities": tf.FixedLenSequenceFeature([], dtype=tf.string),  # entity ids
            "begin_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # mention positions
            "end_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "cand_entities": tf.VarLenFeature(tf.int64),
            "cand_entities_scores": tf.VarLenFeature(tf.float32),
            "cand_entities_labels": tf.VarLenFeature(tf.int64),
            "cand_entities_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "ground_truth": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "begin_gm": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "end_gm": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    context_features = {
        "chunk_id": tf.FixedLenFeature([], dtype=tf.string),
        "words_len": tf.FixedLenFeature([], dtype=tf.int64),
        "spans_len": tf.FixedLenFeature([], dtype=tf.int64),
        "ground_truth_len": tf.FixedLenFeature([], dtype=tf.int64),
        "mask_index": tf.FixedLenFeature([], dtype=tf.int64),
        "mask_ent_id": tf.FixedLenFeature([], dtype=tf.string)
    }
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features=context_features,
        sequence_features=sequence_features)

    ret = [context["chunk_id"], sequence["words"], context["words_len"],
           tf.sparse_tensor_to_dense(sequence["chars"]), sequence["chars_len"],
           sequence["begin_span"], sequence["end_span"], context["spans_len"],
           tf.sparse_tensor_to_dense(sequence["cand_entities"]),
           tf.sparse_tensor_to_dense(sequence["cand_entities_scores"]),
           tf.sparse_tensor_to_dense(sequence["cand_entities_labels"]),
           sequence["cand_entities_len"], sequence["ground_truth"],
           context["ground_truth_len"], sequence["begin_gm"], sequence["end_gm"],
           context["mask_index"], sequence["entities"], context["mask_ent_id"]]
    return ret


def train_input_pipeline(filenames, args):
    padding_entity = "502661"
    if args.pre_training:
        padding_entity = "502661_502661_502661"
    padding_values = tuple(["0", tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            tf.cast(0, tf.float32), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            padding_entity, "502661_502661_502661"])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=args.shuffle_capacity)
    dataset = dataset.padded_batch(args.batch_size, dataset.output_shapes, padding_values)
    return dataset


def test_input_pipeline(filenames, args):
    padding_values = tuple(["0", tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            tf.cast(0, tf.float32), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64), tf.cast(0, tf.int64),
                            "502661_502661_502661", "502661_502661_502661"])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    dataset = dataset.padded_batch(1, dataset.output_shapes, padding_values)
    return dataset
