import argparse
import global_model.reader as reader
import model.config as config
import os
import tensorflow as tf
from evaluation.metrics import Evaluator, metrics_calculation, threshold_calculation
import time
import pickle
import numpy as np
from model.util import load_train_args
from global_model.model import Model


def create_training_pipelines(args):
    folder = config.base_folder + "data/tfrecords/" + args.experiment_name + \
             ("/gmonly_pre_mask/" if args.pre_training else "/gmonly_gt_mask/")
    training_dataset = reader.train_input_pipeline([folder + file for file in args.train_datasets], args)
    return training_dataset


def create_el_ed_pipelines(filenames, args):
    if filenames is None:
        return [], []

    folder = config.base_folder + "data/tfrecords/" + args.experiment_name + \
             ("/gmonly_pre_mask/" if args.pre_training else "/gmonly_gt_mask/")
    test_datasets = []
    for file in filenames:
        test_datasets.append(reader.test_input_pipeline([folder+file], args))

    return test_datasets, filenames


def tensorboard_writers(graph):
    tf_writers = dict()
    tf_writers["train"] = tf.summary.FileWriter(args.summaries_folder + 'train/', graph)
    tf_writers["ed_pr"] = tf.summary.FileWriter(args.summaries_folder + 'ed_pr/')
    tf_writers["ed_re"] = tf.summary.FileWriter(args.summaries_folder + 'ed_re/')
    tf_writers["ed_f1"] = tf.summary.FileWriter(args.summaries_folder + 'ed_f1/')
    return tf_writers


def optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores):
    # based on tp_fp_scores and fn_scores calculate optimal threshold
    tp_fp_scores_labels = sorted(tp_fp_scores_labels)   # low --> high
    fn_scores = sorted(fn_scores)
    tp, fp = 0, 0
    fn_idx = len(fn_scores)    # from [0, fn_idx-1] is fn. [fn_idx, len(fn_scores)) isn't.
    # initially i start with a very high threshold which means I reject everything, hence tp, fp =0
    # and all the gold mentions are fn. so fn = len(fn_scores)
    best_thr = tp_fp_scores_labels[-1][0]+1  # the highest (rightmost) possible threshold + 1 (so everything is rejected)
    best_f1 = -1
    # whatever is on the right or at the position we point is included in the tp, fp
    # whatever is on the left remains to be processed-examined
    tp_fp_idx = len(tp_fp_scores_labels)  # similar to fn_idx
    while tp_fp_idx > 0:  # if we point to 0 then nothing on the left to examine (smaller thresholds)
        # from right to left loop
        tp_fp_idx -= 1
        new_thr, label = tp_fp_scores_labels[tp_fp_idx]
        tp += label
        fp += (1 - label)
        while tp_fp_idx > 0 and tp_fp_scores_labels[tp_fp_idx-1][0] == new_thr:
            tp_fp_idx -= 1
            new_thr, label = tp_fp_scores_labels[tp_fp_idx]
            tp += label
            fp += (1 - label)

        while fn_idx > 0 and fn_scores[fn_idx-1] >= new_thr:  # move left one position
            fn_idx -= 1
        assert( 0 <= tp <= len(tp_fp_scores_labels) and
                0 <= fp <= len(tp_fp_scores_labels) and
                0 <= fn_idx <= len(fn_scores))
        precision = 100 * tp / (tp + fp + 1e-6)
        recall = 100 * tp / (tp + fn_idx + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        assert(0 <= precision <= 100 and 0 <= recall <= 100 and 0 <= f1 <= 100)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = new_thr

    print('Best validation threshold = %.3f with F1=%.1f ' % (best_thr, best_f1))
    return best_thr, best_f1


def validation(model, dataset_handle):
    next_data = model.sess.run([model.next_data], feed_dict={model.input_handle_ph: dataset_handle})
    next_data = next_data[0]
    result_l = [next_data[9], next_data[11], next_data[8], next_data[5], next_data[6], next_data[7],
                next_data[14], next_data[15], next_data[12], next_data[13], next_data[2], next_data[0]]

    # batch_size = 1
    begin_span = np.array(next_data[5])
    end_span = np.array(next_data[6])
    span_len = next_data[7][0]
    entities = next_data[17]
    entities_only = next_data[20]
    default_mask = "502661_502661_502661"

    for k in range(1):
        entities_tmp = np.copy(entities)
        entities_only_tmp = np.copy(entities_only)
        # flag = True
        for i in range(span_len):
            mask_index = np.array([i])
            mask_entities = np.copy(entities_tmp)
            mask_entities_only = np.copy(entities_only_tmp)
            for j in range(begin_span[0][i], end_span[0][i]):
                mask_entities[0][j] = default_mask
            mask_entities_only[i] = default_mask

            pred_scores, local_scores, cand_entities_len, cand_entities = \
                model.sess.run([model.final_scores, model.mask_cand_local_scores,
                                model.mask_cand_entities_len, model.mask_cand_entities],
                               feed_dict={model.dropout: 1,
                                          model.chunk_id: next_data[0],
                                          model.words: next_data[1],
                                          model.words_len: next_data[2],
                                          model.chars: next_data[3],
                                          model.chars_len: next_data[4],
                                          model.begin_span: next_data[5],
                                          model.end_span: next_data[6],
                                          model.spans_len: next_data[7],
                                          model.cand_entities: next_data[8],
                                          model.cand_entities_scores: next_data[9],
                                          model.cand_entities_labels: next_data[10],
                                          model.cand_entities_len: next_data[11],
                                          model.ground_truth: next_data[12],
                                          model.ground_truth_len: next_data[13],
                                          model.begin_gm: next_data[14],
                                          model.end_gm: next_data[15],
                                          model.mask_index: mask_index,
                                          model.entities: mask_entities,
                                          model.cand_local_scores: next_data[18],
                                          model.mask_ent_id: next_data[19],
                                          model.mask_entities_only: mask_entities_only})
            result_l[0][0][i] = pred_scores[0]
            if args.use_local:
                result_l[0][0][i] = 0.4*result_l[0][0][i] + 0.6*local_scores[0]

    return result_l


def validation_loss_calculation(model, iterator, dataset_handle, opt_thr, el_mode, name=""):
    model.sess.run(iterator.initializer)
    evaluator = Evaluator(opt_thr, name=name)
    while True:
        try:
            result_l = validation(model, dataset_handle)
            metrics_calculation(evaluator, *result_l, el_mode)
        except tf.errors.OutOfRangeError:
            print(name)
            micro_f1, macro_f1 = evaluator.print_log_results(model.tf_writers, args.eval_cnt, el_mode)
            break
    return micro_f1, macro_f1


def optimal_thr_calc(model, handles, iterators, el_mode):
    val_datasets = args.ed_val_datasets
    tp_fp_scores_labels = []
    fn_scores = []
    for val_dataset in val_datasets:  # 1, 4
        dataset_handle = handles[val_dataset]
        iterator = iterators[val_dataset]
        model.sess.run(iterator.initializer)
        while True:
            try:
                result_l = validation(model, dataset_handle)
                tp_fp_batch, fn_batch = threshold_calculation(*result_l, el_mode)
                tp_fp_scores_labels.extend(tp_fp_batch)
                fn_scores.extend(fn_batch)
            except tf.errors.OutOfRangeError:
                break
    return optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores)


def compute_ed_el_scores(model, handles, names, iterators, el_mode):
    # first compute the optimal threshold based on validation datasets.
    val_f1 = 0
    if args.hardcoded_thr:
        opt_thr = args.hardcoded_thr
    else:
        opt_thr, val_f1 = optimal_thr_calc(model, handles, iterators, el_mode)

    micro_results = [val_f1]
    macro_results = []
    test_datasets = args.ed_test_datasets
    for i in test_datasets[0:2]:
        test_it = iterators[i]
        test_handle = handles[i]
        test_name = names[i]
        micro_f1, macro_f1 = validation_loss_calculation(model, test_it, test_handle, opt_thr, el_mode=el_mode, name=test_name)
        micro_results.append(micro_f1)
        macro_results.append(macro_f1)
    return micro_results, opt_thr


def compute_other_test_scores(model, handles, names, iterators, opt_thr):
    test_datasets = args.ed_test_datasets
    for i in test_datasets[2:]:
        test_it = iterators[i]
        test_handle = handles[i]
        test_name = names[i]
        validation_loss_calculation(model, test_it, test_handle, opt_thr, el_mode=False, name=test_name)
        return


def ed_el_dataset_handles(datasets, sess):
    test_iterators = []
    test_handles = []
    for dataset in datasets:
        test_iterator = dataset.make_initializable_iterator()
        test_iterators.append(test_iterator)
        test_handles.append(sess.run(test_iterator.string_handle()))
    return test_iterators, test_handles


def train():
    training_dataset = create_training_pipelines(args)
    ed_datasets, ed_names = create_el_ed_pipelines(filenames=args.ed_datasets, args=args)

    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    iterator = tf.contrib.data.Iterator.from_string_handle(
        input_handle_ph, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    model = Model(args, next_element)
    model.build()
    model.input_handle_ph = input_handle_ph

    tf_writers = tensorboard_writers(model.sess.graph)
    model.tf_writers = tf_writers  # for accessing convenience

    with model.sess as sess:
        training_iterator = training_dataset.make_one_shot_iterator()
        training_handle = sess.run(training_iterator.string_handle())
        ed_iterators, ed_handles = ed_el_dataset_handles(ed_datasets, sess)

        # Loop forever, alternating between training and validation.
        best_ed_score = 0
        termination_ed_score = 0
        nepoch_no_imprv = 0  # for early stopping
        train_step = 0
        print("start training!")
        while True:
            total_train_loss = 0
            wall_start = time.time()
            while ((time.time() - wall_start) / 60) <= args.evaluation_minutes:
                train_step += 1
                next_data = sess.run([model.next_data], feed_dict={input_handle_ph: training_handle})
                next_data = next_data[0]
                _, loss = sess.run([model.train_op, model.loss],
                                   feed_dict={model.dropout: args.dropout,
                                              model.lr: model.args.lr,
                                              model.chunk_id: next_data[0],
                                              model.words: next_data[1],
                                              model.words_len: next_data[2],
                                              model.chars: next_data[3],
                                              model.chars_len: next_data[4],
                                              model.begin_span: next_data[5],
                                              model.end_span: next_data[6],
                                              model.spans_len: next_data[7],
                                              model.cand_entities: next_data[8],
                                              model.cand_entities_scores: next_data[9],
                                              model.cand_entities_labels: next_data[10],
                                              model.cand_entities_len: next_data[11],
                                              model.ground_truth: next_data[12],
                                              model.ground_truth_len: next_data[13],
                                              model.begin_gm: next_data[14],
                                              model.end_gm: next_data[15],
                                              model.mask_index: next_data[16],
                                              model.entities: next_data[17],
                                              model.cand_local_scores: next_data[18],
                                              model.mask_ent_id: next_data[19],
                                              model.mask_entities_only: next_data[20]})
                total_train_loss += loss
                if train_step % 100 == 0:
                    print("train_step =", train_step, "train_loss =", loss, flush=True)

            args.eval_cnt += 1
            summary = tf.Summary(value=[tf.Summary.Value(tag="total_train_loss", simple_value=total_train_loss)])
            tf_writers["train"].add_summary(summary, args.eval_cnt)

            print("args.eval_cnt = ", args.eval_cnt, flush=True)
            wall_start = time.time()
            comparison_ed_score = -0.1
            if ed_names:
                print("Evaluating ED datasets")
                ed_scores, opt_thr = compute_ed_el_scores(model, ed_handles, ed_names, ed_iterators, el_mode=False)
                comparison_ed_score = np.mean(np.array(ed_scores)[args.ed_val_datasets])
            print("Evaluation duration in minutes: ", (time.time() - wall_start) / 60)

            if model.args.lr_decay > 0:
                model.args.lr *= model.args.lr_decay  # decay learning rate
            text = ""
            best_ed_flag = False

            if comparison_ed_score >= best_ed_score + 0.1:  # args.improvement_threshold:
                text = "- new best ED score!" + " prev_best= " + str(best_ed_score) + \
                       " new_best= " + str(comparison_ed_score)
                best_ed_flag = True
                best_ed_score = comparison_ed_score

            if best_ed_flag:  # keep checkpoint
                print(text)
                if args.nocheckpoints is False:
                    model.save_session(args.eval_cnt, best_ed_flag, False)

            # check for termination now.
            if comparison_ed_score >= termination_ed_score + args.improvement_threshold:
                print("significant improvement. reset termination counter")
                termination_ed_score = comparison_ed_score
                nepoch_no_imprv = 0
                if args.eval_cnt > 5:
                    print("Evaluating ED other test datasets")
                    compute_other_test_scores(model, ed_handles, ed_names, ed_iterators, opt_thr)
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= args.nepoch_no_imprv:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    terminate()
                    break
        print("finish training!")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="alldatasets_perparagr",  # "standard",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", default=None,
                        help="under folder data/tfrecords/")
    parser.add_argument("--shuffle_capacity", type=int, default=500)
    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--nepoch_no_imprv", type=int, default=5)
    parser.add_argument("--improvement_threshold", type=float, default=0.3, help="if improvement less than this then"
                        "it is considered not significant and we have early stopping.")
    parser.add_argument("--clip", type=int, default=-1, help="if negative then no clipping")
    parser.add_argument("--lr_decay", type=float, default=-1.0, help="if negative then no decay")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_method", default="adam")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_ent_vecs", dest='train_ent_vecs', action='store_true')
    parser.add_argument("--no_train_ent_vecs", dest='train_ent_vecs', action='store_false')
    parser.set_defaults(train_ent_vecs=False)

    parser.add_argument("--steps_before_evaluation", type=int, default=10000)
    parser.add_argument("--evaluation_minutes", type=int, default=15, help="every this number of minutes pause"
                                                                           " training and run an evaluation epoch")
    parser.add_argument("--dim_char", type=int, default=100)
    parser.add_argument("--hidden_size_char", type=int, default=100, help="lstm on chars")
    parser.add_argument("--hidden_size_lstm", type=int, default=300, help="lstm on word embeddings")

    parser.add_argument("--use_local", dest="use_local", action='store_true')
    parser.add_argument("--no_use_local", dest="use_local", action='store_false')
    parser.set_defaults(use_local=False)

    parser.add_argument("--model_heads_from_bilstm", type=bool, default=False,
                        help="use the bilstm vectors for the head instead of the word embeddings")
    parser.add_argument("--span_boundaries_from_wordemb", type=bool, default=False, help="instead of using the "
                        "output of contextual bilstm for start and end of span we use word+char emb")
    parser.add_argument("--span_emb", default="boundaries_head", help="boundaries for start and end, and head")

    parser.add_argument("--max_mention_width", type=int, default=10)
    parser.add_argument("--use_features", type=bool, default=False, help="like mention width")
    parser.add_argument("--feature_size", type=int, default=20)   # each width is represented by a vector of that size

    parser.add_argument("--ent_vecs_regularization", default="l2dropout", help="'no', ""'dropout', 'l2', 'l2dropout'")

    parser.add_argument("--span_emb_ffnn", default="0_0", help="int_int  the first int"
                        "indicates the number of hidden layers and the second the hidden size"
                        "so 2_100 means 2 hidden layers of width 100 and then projection to output size"
                        ". 0_0 means just projecting without hidden layers")
    parser.add_argument("--final_score_ffnn", default="1_100", help="int_int  look span_emb_ffnn")

    parser.add_argument("--gamma_thr", type=float, default=0.2)

    parser.add_argument("--nocheckpoints", type=bool, default=False)
    parser.add_argument("--checkpoints_num", type=int, default=1, help="maximum number of checkpoints to keep")

    parser.add_argument("--ed_datasets", default="")
    parser.add_argument("--ed_val_datasets", default="0", help="based on these datasets pick the optimal"
                                                               "gamma thr and also consider early stopping")
    parser.add_argument("--ed_test_datasets", default="1")

    parser.add_argument("--el_datasets", default="")

    parser.add_argument("--train_datasets", default="aida_train")

    parser.add_argument("--continue_training", type=bool, default=False,
                        help="if true then just restore the previous command line"
                             "arguments and continue the training in exactly the"
                             "same way. so only the experiment_name and "
                             "training_name are used from here. Retrieve values from"
                             "latest checkpoint.")
    parser.add_argument("--onleohnard", type=bool, default=False)

    parser.add_argument("--comment", default="", help="put any comment here that describes your experiment"
                                                      ", for logging purposes only.")

    parser.add_argument("--pre_training", dest='pre_training', action='store_true')
    parser.add_argument("--no_pre_training", dest='pre_training', action='store_false')
    parser.set_defaults(pre_training=False)

    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    parser.add_argument("--nn_components", default="pem_lstm", help="each option is one scalar, then these are fed to"
                        "the final ffnn and we have the final score. choose any combination you want: e.g"
                        "pem_lstm_attention_global, pem_attention, lstm_attention, pem_lstm_global, etc")
    parser.add_argument("--attention_K", type=int, default=100, help="K from left and K from right, in total 2K")
    parser.add_argument("--attention_R", type=int, default=30, help="hard attention")
    parser.add_argument("--attention_use_AB", type=bool, default=False)
    parser.add_argument("--attention_on_lstm", type=bool, default=False, help="instead of using attention on"
                        "original pretrained word embedding. use it on vectors or lstm, "
                        "needs also projection now the context vector x_c to 300 dimensions")
    parser.add_argument("--attention_ent_vecs_no_regularization", type=bool, default=False)
    parser.add_argument("--attention_retricted_num_of_entities", type=int, default=None,
                        help="instead of using 30 entities for creating the context vector we use only"
                             "the top x number of entities for reducing noise.")
    parser.add_argument("--global_thr", type=float, default=0.1)   # 0.0, 0.05, -0.05, 0.2
    parser.add_argument("--global_mask_scale_each_mention_voters_to_one", type=bool, default=False)
    parser.add_argument("--global_topk", type=int, default=None)
    parser.add_argument("--global_gmask_based_on_localscore", type=bool, default=False)   # new
    parser.add_argument("--global_topkthr", type=float, default=None)   # 0.0, 0.05, -0.05, 0.2
    parser.add_argument("--global_score_ffnn", default="1_100", help="int_int  look span_emb_ffnn")
    parser.add_argument("--global_one_loss", type=bool, default=False)
    parser.add_argument("--global_norm_or_mean", default="norm")
    parser.add_argument("--global_topkfromallspans", type=int, default=None)
    parser.add_argument("--global_topkfromallspans_onlypositive", type=bool, default=False)
    parser.add_argument("--global_gmask_unambigious", type=bool, default=False)

    parser.add_argument("--hardcoded_thr", type=float, default=None, help="if this is specified then we don't calculate"
                        "optimal threshold based on the dev dataset but use this one.")
    parser.add_argument("--ffnn_dropout", dest="ffnn_dropout", action='store_true')
    parser.add_argument("--no_ffnn_dropout", dest="ffnn_dropout", action='store_false')
    parser.set_defaults(ffnn_dropout=True)
    parser.add_argument("--ffnn_l2maxnorm", type=float, default=None, help="if positive"
                        " then bound the Frobenius norm <= value for the weight tensor of the "
                        "hidden layers and the output layer of the FFNNs")
    parser.add_argument("--ffnn_l2maxnorm_onlyhiddenlayers", type=bool, default=False)

    parser.add_argument("--cand_ent_num_restriction", type=int, default=None, help="for reducing memory usage and"
                        "avoiding OOM errors in big NN I can reduce the number of candidate ent for each span")
    parser.add_argument("--no_p_e_m_usage", type=bool, default=False, help="use similarity score instead of "
                                                                           "final score for prediction")
    parser.add_argument("--pem_without_log", type=bool, default=False)
    parser.add_argument("--pem_buckets_boundaries", default=None,
                        help="example: 0.03_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.99")
    # the following two command line arguments
    parser.add_argument("--gpem_without_log", type=bool, default=False)
    parser.add_argument("--gpem_buckets_boundaries", default=None,
                        help="example: 0.03_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.99")
    parser.add_argument("--stage2_nn_components", default="local_global", help="each option is one scalar, "
                        "then these are fed to the final ffnn and we have the final score. "
                        "choose any combination you want: e.g pem_local_global, pem_global, local_global, global, etc")
    parser.add_argument("--ablations", type=bool, default=False)
    args = parser.parse_args()

    if args.training_name is None:
        from datetime import datetime
        args.training_name = "{:%d_%m_%Y____%H_%M}".format(datetime.now())

    temp = "local_" if args.pre_training else ""
    args.output_folder = config.base_folder+"data/tfrecords/" + \
                         args.experiment_name+"/{}training_folder/".format(temp)+\
                         args.training_name+"/"

    if args.continue_training:
        print("continue training...")
        train_args = load_train_args(args.output_folder, "train_continue")
        return train_args
    args.running_mode = "train"  # "evaluate"  "ensemble_eval"  "gerbil"

    if os.path.exists(args.output_folder) and not args.continue_training:
        print("!!!!!!!!!!!!!!\n"
              "experiment: ", args.output_folder, "already exists and args.continue_training = False. "
              "folder will be deleted in 20 seconds. Press CTRL+C to prevent it.")
        time.sleep(1)
        import shutil
        shutil.rmtree(args.output_folder)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.checkpoints_folder = args.output_folder + "checkpoints/"
    if args.onleohnard:
        args.checkpoints_folder = "/cluster/home/nkolitsa/checkpoints_folder/"+\
            args.experiment_name + "/" + args.training_name + "/"

    args.summaries_folder = args.output_folder + "summaries/"
    if not os.path.exists(args.summaries_folder):
        os.makedirs(args.summaries_folder)

    args.ed_datasets = args.ed_datasets.split('_z_') if args.ed_datasets != "" else None
    args.el_datasets = args.el_datasets.split('_z_') if args.el_datasets != "" else None
    args.train_datasets = args.train_datasets.split('_z_') if args.train_datasets != "" else None

    args.ed_val_datasets = [int(x) for x in args.ed_val_datasets.split('_')]
    args.ed_test_datasets = [int(x) for x in args.ed_test_datasets.split('_')]

    args.span_emb_ffnn = [int(x) for x in args.span_emb_ffnn.split('_')]
    args.final_score_ffnn = [int(x) for x in args.final_score_ffnn.split('_')]
    args.global_score_ffnn = [int(x) for x in args.global_score_ffnn.split('_')]

    args.eval_cnt = 0
    args.zero = 1e-6

    if args.pem_buckets_boundaries:
        args.pem_buckets_boundaries = [float(x) for x in args.pem_buckets_boundaries.split('_')]
    if args.gpem_buckets_boundaries:
        args.gpem_buckets_boundaries = [float(x) for x in args.gpem_buckets_boundaries.split('_')]

    return args


def log_args(filepath):
    with open(filepath, "w") as fout:
        attrs = vars(args)
        fout.write('\n'.join("%s: %s" % item for item in attrs.items()))

    with open(args.output_folder+"train_args.pickle", 'wb') as handle:
        pickle.dump(args, handle)


def terminate():
    tee.close()
    with open(args.output_folder+"train_args.pickle", 'wb') as handle:
        pickle.dump(args, handle)


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    log_args(args.output_folder+"train_args.txt")
    from model.util import Tee
    tee = Tee(args.output_folder+'log.txt', 'a')
    try:
        train()
    except KeyboardInterrupt:
        terminate()
