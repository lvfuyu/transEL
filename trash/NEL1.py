import json
import model.config as config
from gerbil.nn_processing import NNProcessing
from model.util import load_train_args
from gerbil.build_entity_universe import BuildEntityUniverse
import argparse

def read_json(post_data):
    data = json.loads(post_data)
    #print("received data:", data)
    text = data["text"]
    spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
    return text, spans

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="per_document_no_wikidump",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", default="doc_fixed_nowiki_evecsl2dropout")
    parser.add_argument("--all_spans_training", type=bool, default=False)
    parser.add_argument("--el_mode", dest='el_mode', action='store_true')
    parser.add_argument("--ed_mode", dest='el_mode', action='store_false')
    parser.set_defaults(el_mode=True)

    parser.add_argument("--running_mode", default=None, help="el_mode or ed_mode, so"
                                "we can restore an ed_mode model and run it for el")

    parser.add_argument("--lowercase_spans_pem", type=bool, default=False)

    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    # those are for building the entity set
    parser.add_argument("--build_entity_universe", type=bool, default=False)
    parser.add_argument("--hardcoded_thr", type=float, default=None, help="0, 0.2")
    parser.add_argument("--el_with_stanfordner_and_our_ed", type=bool, default=False)

    parser.add_argument("--persons_coreference", type=bool, default=False)
    parser.add_argument("--persons_coreference_merge", type=bool, default=False)

    args = parser.parse_args()
    if args.persons_coreference_merge:
        args.persons_coreference = True
    print(args)
    if args.build_entity_universe:
        return args, None

    temp = "all_spans_" if args.all_spans_training else ""
    args.experiment_folder = config.base_folder+"data/tfrecords/" + args.experiment_name+"/"

    args.output_folder = config.base_folder+"data/tfrecords/" + \
                         args.experiment_name+"/{}training_folder/".format(temp) + \
                         args.training_name+"/"

    train_args = load_train_args(args.output_folder, "gerbil")
    train_args.entity_extension = args.entity_extension

    print(train_args)
    return args, train_args


if __name__ == "__main__":
    args, train_args = _parse_args()
    nnprocessing = NNProcessing(train_args, args)
    post_data = '{ "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": []  }'
    response = nnprocessing.process(*read_json(post_data))
    print(response)


