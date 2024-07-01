# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for MMMU Aggregator Component."""

import argparse

from aml_benchmark.utils.io import read_json_data, save_json_to_file
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.exceptions import (
    swallow_all_exceptions,
)


SHORT_NAMES = [
    "accounting",
    "agriculture",
    "arch_eng",
    "art",
    "art_theory",
    "med_sci",
    "biology",
    "chem",
    "clin_med",
    "comp_sci",
    "design",
    "diag",
    "econ",
    "electro",
    "energy",
    "fin",
    "geo",
    "hist",
    "lit",
    "mngm",
    "mrkt",
    "mat",
    "math",
    "me",
    "mus",
    "pharm",
    "phys",
    "psych",
    "pub_health",
    "soc",
]
LONG_NAME_BY_SHORT_NAME = {
    "accounting": "Accounting",
    "agriculture": "Agriculture",
    "arch_eng": "Architecture_and_Engineering",
    "art": "Art",
    "art_theory": "Art_Theory",
    "med_sci": "Basic_Medical_Science",
    "biology": "Biology",
    "chem": "Chemistry",
    "clin_med": "Clinical_Medicine",
    "comp_sci": "Computer_Science",
    "design": "Design",
    "diag": "Diagnostics_and_Laboratory_Medicine",
    "econ": "Economics",
    "electro": "Electronics",
    "energy": "Energy_and_Power",
    "fin": "Finance",
    "geo": "Geography",
    "hist": "History",
    "lit": "Literature",
    "mngm": "Manage",
    "mrkt": "Marketing",
    "mat": "Materials",
    "math": "Math",
    "me": "Mechanical_Engineering",
    "mus": "Music",
    "pharm": "Pharmacy",
    "phys": "Physics",
    "psych": "Psychology",
    "pub_health": "Public_Health",
    "soc": "Sociology",
}
DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    for short_name in SHORT_NAMES:
       parser.add_argument(
            "--evaluation_result_{short_name}",
            type=str,
            default=None,
        )
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


@swallow_all_exceptions(logger)
def main(args) -> None:
    """
    Entry function for MMMU Aggregator.

    ???
    :return: None
    """
    quality_metrics = {}
    for short_name in SHORT_NAMES:
        quality_metrics[short_name] = read_json_data(getattr(args, "evaluation_result_{short_name}"))
    print("x1", quality_metrics)

    sum_num_instances = sum([m["num_instances"] for m in quality_metrics.values()])
    sum_accuracies = sum([m["accuracy"] for m in quality_metrics.values()])
    print("x2", float(sum_accuracies) / sum_num_instances)

    ln2sn = {v: k for k, v in LONG_NAME_BY_SHORT_NAME}
    for domain, long_names in DOMAIN_CAT2SUB_CAT.items():
        metrics = [quality_metrics[ln2sn[long_name]] for long_name in long_names]
        sum_num_instances = sum([m["num_instances"] for m in metrics])
        sum_accuracies = sum([m["accuracy"] for m in metrics])
        print("x3", domain, float(sum_accuracies) / sum_num_instances)


if __name__ == "__main__":
    args = parse_args()
    main(
        evaluation_result_accounting=args.evaluation_result_accounting,
    )
