"""File for the zero shot component."""
from argparse import ArgumentParser
from run_utils import (
    current_run,
    log_results
)
from azureml.rag.utils.logging import enable_stdout_logging, enable_appinsights_logging, track_activity
from constants import ALL_TASKS, TaskTypes
from tasks.task_utils import get_task
from utils import (
    read_data,
    openai_init,
    make_and_log_exception,
    CUSTOM_DIMENSIONS
)
from logging_utilities import (
    get_logger,
    log_info
)
from error_definitions import (
    AutoPromptInternalError,
    InvalidTaskType,
    OpenAIInitError,
    MetricsLoggingError
)
from exceptions import (
    AutoPromptException,
    ArgumentValidationException,
)

import pandas as pd
import os
import constants
import json
from azureml.exceptions import AzureMLException

logger = get_logger("autoprompt")


def main():
    """File Main Entry Point."""
    parser = ArgumentParser()
    parser.add_argument("--data_file_name",
                        default="QAGenerationData.jsonl",
                        type=str,
                        required=False,
                        help="The input data file name within folder directory")
    parser.add_argument("--dev_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input test data in Json Lines format.")
    parser.add_argument("--test_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input test data in Json Lines format.")
    parser.add_argument("--prompt",
                        type=str,
                        required=False,
                        help="Prompt to send to the model")
    parser.add_argument("--task_type",
                        type=str,
                        required=True,
                        choices=ALL_TASKS,
                        help="Task Type")
    parser.add_argument("--text_keys",
                        type=str,
                        required=True,
                        help="The name of keys containing input texts")
    parser.add_argument("--label_key",
                        type=str,
                        required=True,
                        help="The name of the key containing labels")
    parser.add_argument("--predictions",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_type",
                        default="zero_shot",
                        type=str,
                        required=False,
                        choices=["zero_shot", "few_shot"])
    parser.add_argument("--primary_metric",
                        default="f1_score",
                        type=str,
                        required=True,
                        choices=constants.ALL_METRICS)
    parser.add_argument("--temperature",
                        default=0.0,
                        type=float,
                        required=False)
    parser.add_argument("--max_tokens",
                        default=16,
                        type=int,
                        required=False)
    parser.add_argument("--logprobs",
                        default=0,
                        type=int,
                        required=False)
    parser.add_argument("--choices_key",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--n_prompts",
                        default=10,
                        type=int,
                        required=False)
    parser.add_argument("--llm_config",
                        type=str,
                        default='{"type": "azure_open_ai", "model_name": "text-davinci-002", "deployment_name": "text-davinci-002"}',
                        required=False)
    parser.add_argument("--openai_api_version",
                        type=str,
                        default="2022-12-01",
                        required=True)
    parser.add_argument("--openai_api_type",
                        type=str,
                        default="azure",
                        required=True)
    parser.add_argument("--best_of",
                        default=100,
                        type=int,
                        required=False)
    parser.add_argument("--top_k",
                        default=3,
                        type=int,
                        required=False)

    args = parser.parse_args()
    data_file_name = args.data_file_name
    dev_data = args.dev_data
    test_data = args.test_data
    meta_prompts_file = args.prompt
    task_type = args.task_type
    text_keys = [i.strip() for i in args.text_keys.split(",")]
    label_key = args.label_key
    predictions_file = args.predictions
    choices_key = args.choices_key
    n_prompts = args.n_prompts
    best_of = args.best_of
    primary_metric = args.primary_metric
    top_k = args.top_k

    if task_type not in constants.ALL_TASKS:
        make_and_log_exception(
            error_cls=InvalidTaskType,
            exception_cls=ArgumentValidationException,
            inner_exception=ValueError("Invalid Task Type."),
            logger=logger,
            activity_logger=main_activity_logger,
            message_params={"task_type": task_type}
        )

    # If input is a directory rather than file, find the file with QA data in it
    if (data_file_name is None):
        data_file_name = "QAGenerationData.jsonl"
    if (os.path.isdir(test_data)):
        log_info(logger, "test_data path", CUSTOM_DIMENSIONS)
        test_data = os.path.join(test_data, data_file_name)
        log_info(logger, test_data, CUSTOM_DIMENSIONS)
    if (os.path.isdir(dev_data)):
        log_info(logger, "dev_data path", CUSTOM_DIMENSIONS)
        dev_data = os.path.join(dev_data, data_file_name)
        log_info(logger, dev_data, CUSTOM_DIMENSIONS)

    llm_config = json.loads(args.llm_config)
    openai_init_params = {
        "openai_api_type": args.openai_api_type,
        "openai_api_version": args.openai_api_version
    }
    try:
        openai_params = openai_init(llm_config=llm_config, **openai_init_params)
    except Exception as e:
        make_and_log_exception(
            error_cls=OpenAIInitError,
            exception_cls=AutoPromptException,
            logger=logger,
            activity_logger=main_activity_logger,
            inner_exception=e
        )

    meta_prompts = []
    if meta_prompts_file:
        log_info(logger, "Reading User Meta Prompts", CUSTOM_DIMENSIONS)
        with open(meta_prompts_file, "r") as f:
            meta_prompts = json.load(f)
        if len(meta_prompts) > 0 and task_type == TaskTypes.abstractive:
            for i in range(len(meta_prompts)):
                meta_prompts[i] = '"'+meta_prompts[i]+'"'
    else:
        log_info(logger, "No Meta prompts passed by user. Using Default meta prompts", CUSTOM_DIMENSIONS)

    log_info(logger, "Meta Prompts: "+str(meta_prompts), CUSTOM_DIMENSIONS)
    df_dev, y_true_dev, df_test, y_true_test, num_choices = read_data(task_type,
                                                                      dev_data,
                                                                      test_data,
                                                                      text_keys,
                                                                      label_key,
                                                                      choices_key)
    dev_data_df = pd.read_json(dev_data, lines=True, dtype=False)
    val_data_df = pd.read_json(test_data, lines=True, dtype=False)

    keyword_args = {
        "openai_params": openai_params,
        "llm_config": llm_config,
        "questions": dev_data_df[text_keys[0]],
        "valid_questions": val_data_df[text_keys[0]]
    }
    if task_type == TaskTypes.abstractive:
        keyword_args["contexts"] = dev_data_df[choices_key]
        keyword_args["valid_contexts"] = val_data_df[choices_key]
    task_obj = get_task(task_type, meta_prompts, num_choices, n_prompts, best_of, primary_metric)
    task_results = task_obj.find_best_prompt(df_dev, y_true_dev, df_test, y_true_test, text_keys, **keyword_args)

    with track_activity(logger,
                        constants.Activities.LOG_RESULTS,
                        custom_dimensions=CUSTOM_DIMENSIONS) as metrics_logging_activity:
        try:
            log_info(logger, "Logging Results", CUSTOM_DIMENSIONS)
            log_results(task_results, y_true_test, top_k, predictions_file)
        except Exception as e:
            make_and_log_exception(
                error_cls=MetricsLoggingError,
                exception_cls=AutoPromptException,
                logger=logger,
                activity_logger=metrics_logging_activity,
                inner_exception=e,
                message_params={"error": repr(e)}
            )
    current_run.complete()


if __name__ == "__main__":
    enable_appinsights_logging()
    enable_stdout_logging()
    with track_activity(logger, constants.Activities.MAIN,
                        custom_dimensions=CUSTOM_DIMENSIONS) as main_activity_logger:
        try:
            main()
        except AzureMLException as aml_exception:
            raise aml_exception
        except Exception as e:
            make_and_log_exception(
                error_cls=AutoPromptInternalError,
                exception_cls=AutoPromptException,
                logger=logger,
                inner_exception=e,
                activity_logger=main_activity_logger,
                message_params={"error": repr(e)}
            )
