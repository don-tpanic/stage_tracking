import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import argparse
import numpy as np
import pandas as pd
import utils


def read_data(data_fpath, columns):
    pass


def train_eval_split(raw_data_fpath, n_evals=5):
    pass


def testset_loading(raw_testdata_fpath):
    pass


def merge_convo_rounds(data):
    pass


def pair_convo_rounds_next_stage(json_collector):
    pass


if __name__ == '__main__':
    columns = []
    send_receive_values = []
    raw_data_fpath = "data/raw.csv"
    raw_testdata_fpath = "data/raw_test.csv"

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_version", type=str)
    config = utils.load_config(f"config_{argparser.parse_args().config_version}")
    convo_format = config['convo_format']
    data_path = f"data/{convo_format}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Split raw data into 25:5 for train and eval
    train_data, eval_data = train_eval_split(raw_data_fpath, n_evals=5)
    # Load testset
    test_data = testset_loading(raw_testdata_fpath)

    # Merge convo into rounds
    train_data = merge_convo_rounds(train_data)
    eval_data = merge_convo_rounds(eval_data) 
    test_data = merge_convo_rounds(test_data)

    if convo_format == "paired_next_stage":
        # Create paired dataset for training and testing
        train_data = pair_convo_rounds_next_stage(train_data)
        eval_data = pair_convo_rounds_next_stage(eval_data)
        test_data = pair_convo_rounds_next_stage(test_data)

        # Save to json
        with open(f"{data_path}/train.json", "w") as f:
            json.dump(train_data, f, ensure_ascii=False)
        with open(f"{data_path}/eval.json", "w") as f:
            json.dump(eval_data, f, ensure_ascii=False)
        with open(f"{data_path}/test.json", "w") as f:
            json.dump(test_data, f, ensure_ascii=False)
