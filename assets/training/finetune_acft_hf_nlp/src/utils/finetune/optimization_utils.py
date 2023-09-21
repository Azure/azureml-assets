# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from azureml.acft.contrib.hf.nlp.constants.constants import Tasks
from argparse import Namespace


def should_apply_ort(args: Namespace, logger):
    if args.apply_ort and args.task_name in (Tasks.SUMMARIZATION, Tasks.TRANSLATION):
        logger.warning("Enabling ORT has a breaking change with summarization and translation tasks "
                       "so diabling ORT for SUMMARIZATION and TRANSLATION tasks")
        return False
    return args.apply_ort
