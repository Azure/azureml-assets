# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import pandas as pd
from argparse import ArgumentParser


def _get_intersection(x, y):
    result = [value for value in x if value in y]
    return result


def _get_sample_queries(samples):
    result = []
    for sample in samples:
        result.append(sample["Question"])
    return result


def action_statistic_check(action_path, debugging):
    folder = os.path.abspath(os.path.dirname(__file__))
    golden_df = pd.read_csv(os.path.join(folder, "resources/golden_set.csv"))
    golden_set_bad_queries = golden_df[golden_df["IsNegativeSample"] == "T"]["prompt"].to_list()
    print("========== True Bad Query Count ==========")
    print(len(golden_set_bad_queries))

    with open(os.path.join(folder, action_path)) as f:
        action_data = json.load(f)
        good_samples = _get_sample_queries(action_data["PositiveSamples"])
        bad_samples = _get_sample_queries(action_data["NegativeSamples"])
        print("========== Good Samples Count ==========")
        print(len(good_samples))
        print("========== Bad Samples Count ==========")
        print(len(bad_samples))

        true_positive = _get_intersection(golden_set_bad_queries, bad_samples)
        precision = float(len(true_positive)/len(bad_samples))*100
        print("========== Precision ==========")
        print(f"{round(precision, 2)}%")

        recall = float(len(true_positive)/len(golden_set_bad_queries))*100
        print("========== Recall ==========")
        print(f"{round(recall, 2)}%")

        if debugging:
            print("\n")
            false_positive = [q for q in bad_samples if q not in golden_set_bad_queries]
            print("========== False Positive (should be good query but in bad samples) ==========")
            print(false_positive)
            print("\n")

            missed_positive = [q for q in golden_set_bad_queries if q not in bad_samples]
            print("========== Missed Positive (should be bad query but did not generated in action) ==========")
            print(missed_positive)
            print("\n")

            false_negative = [q for q in good_samples if q in golden_set_bad_queries]
            print("========== False Negative (should be bad query but in good samples) ==========")
            print(false_negative)
            print("\n")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-f", "--file", help="file path of action json file")
    argparser.add_argument("-d", "--debug", action='store_true', help="debug flag")
    args = argparser.parse_args()
    action_statistic_check(args.file, args.debug)
