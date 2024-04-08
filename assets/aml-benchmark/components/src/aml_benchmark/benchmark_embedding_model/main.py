# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Benchmark Embedding Model Component."""

import os
from typing import Optional, Tuple, List
import argparse

from azureml._common._error_definition.azureml_error import AzureMLError
from mteb import MTEB, MTEB_MAIN_EN
from mteb.abstasks.TaskMetadata import TASK_TYPE

from .utils.constants import Preset, DeploymentType
from .deployments.deployment_factory import DeploymentFactory
from ..utils.logging import get_logger, log_mlflow_params
from ..utils.exceptions import swallow_all_exceptions, BenchmarkValidationException
from ..utils.error_definitions import BenchmarkValidationError


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--endpoint_url",
        type=str,
        required=False,
        help="For AOAI, the base endpoint url. For OAI, this will be ignored.",
        default=None,
    )
    parser.add_argument(
        "--deployment_type",
        type=str,
        required=True,
        choices=[member.value for member in DeploymentType],
        help="Choose from one of the deployment types: `AOAI`, `OAI`."
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        required=True,
        help="For AOAI, the deployment name. For OAI, the model name."
    )
    parser.add_argument(
        "--connections_name",
        type=str,
        required=True,
        help="Used for authenticating endpoint."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=False,
        help="Comma separated string denoting the tasks to benchmark the model on.",
        default=None,
    )
    parser.add_argument(
        "--task_types",
        type=str,
        required=False,
        help=(
            "Comma separated string denoting the task type to benchmark the model on. Choose from "
            "the following task types: BitextMining, Classification, Clustering, PairClassification, Reranking, "
            "Retrieval, STS, Summarization"
        ),
        default=None,
    )
    parser.add_argument(
        "--task_langs",
        type=str,
        required=False,
        help="Comma separated string denoting the task languages to benchmark the model on.",
        default=None,
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=False,
        choices=[member.value for member in Preset] + ["None"],
        help="Choose from one of the presets for benchmarking: `mteb_main_en`. Default is `None`.",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help=(
            "Number of texts to send in a single batch. Defaults to 32. Will be automatically reduced "
            "if the batch does not fit in the model context."
        ),
        default=32,
    )
    parser.add_argument(
        "--output_metrics_dir", type=str, required=True, help="Directory where the benchmark metrics will be saved."
    )

    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def _validate_args(
    output_metrics_dir: str,
    tasks: Optional[str] = None,
    task_types: Optional[str] = None,
    task_langs: Optional[str] = None,
    preset: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]], Optional[str]]:
    """
    Validate arguments.

    :param output_metrics_dir: Directory where the benchmark metrics will be saved.
    :param tasks: Comma separated string denoting the tasks to benchmark the model on.
    :param task_types: Comma separated string denoting the task type to benchmark the model on.
    :param task_langs: Comma separated string denoting the task languages to benchmark the model on.
    :param preset: Choose from one of the presets for benchmarking: `None` or `mteb_main_en`.
    :return: Tuple of tasks, task_types, task_langs, preset.
    """
    if not os.path.exists(output_metrics_dir):
        os.makedirs(output_metrics_dir)

    if not tasks and not task_types and not task_langs and not preset:
        mssg = "Either `tasks` or `task_types` or `task_langs` or `preset` must be supplied."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    if preset and preset != "None":
        logger.info(f"Using preset: {preset}. Arguments `tasks`, `task_types`, `task_langs` will be ignored.")
        if preset == Preset.MTEB_MAIN_EN.value:
            tasks = MTEB_MAIN_EN
            task_langs = ["en"]
            task_types = None
        else:
            mssg = f"Invalid preset provided: {preset}. Valid presets: {[member.value for member in Preset]}."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
    else:
        preset = None
        if tasks and task_types:
            logger.warning("Both `tasks` and `task_types` are provided. `task_types` will be ignored.")
            task_types = None

        if tasks:
            tasks = [task.strip() for task in tasks.split(",")]

        if task_types:
            _validated_task_types = []
            for task_type in task_types.split(","):
                task_type = task_type.strip()
                if task_type in TASK_TYPE.__args__:
                    _validated_task_types.append(task_type)
                else:
                    logger.warning(f"Ignoring invalid task type: {task_type}.")
            if not _validated_task_types:
                mssg = f"None of the provided task types are valid: Provided: {task_types}. Valid: {TASK_TYPE.__args__}."
                raise BenchmarkValidationException._with_error(
                    AzureMLError.create(BenchmarkValidationError, error_details=mssg)
                )
            task_types = _validated_task_types

        if task_langs:
            task_langs = [lang.strip() for lang in task_langs.split(",")]

    return tasks, task_types, task_langs, preset


@swallow_all_exceptions(logger)
def main(
    output_metrics_dir: str,
    deployment_type: str,
    deployment_name: str,
    connections_name: str,
    endpoint_url: Optional[str] = None,
    tasks: Optional[str] = None,
    task_types: Optional[str] = None,
    task_langs: Optional[str] = None,
    preset: Optional[str] = None,
    batch_size: int = 32,
) -> None:
    """
    Entry function for Benchmark Embedding Model Component.

    Either `tasks` or `task_types` or `task_langs` or `preset` must be supplied.

    :param output_metrics_dir: Directory where the benchmark metrics will be saved.
    :param deployment_type: Choose from one of the deployment types: `AOAI`, `OAI`.
    :param deployment_name: For AOAI, the deployment name. For OAI, the model name.
    :param connections_name: Used for authenticating endpoint.
    :param endpoint_url: For AOAI, the base endpoint url. For OAI, this will be ignored.
    :param tasks: Comma separated string denoting the tasks to benchmark the model on.
    :param task_types: Comma separated string denoting the task type to benchmark the model on. Choose from the
        following task types: classification, clustering, pair_classification, reranking, retrieval, sts,
        summarization, all.
    :param task_langs: Comma separated string denoting the task languages to benchmark the model on.
    :param preset: Choose from one of the presets for benchmarking: `None` or `mteb_main_en`. Default is `None`.
    :param batch_size: Number of texts to send in a single batch. Defaults to 32. Will be automatically reduced
        if the batch does not fit in the model context.
    :return: None
    """
    # Argument Validation
    validated_tasks, validated_task_types, validated_task_langs, preset = _validate_args(
        output_metrics_dir=output_metrics_dir,
        tasks=tasks,
        task_types=task_types,
        task_langs=task_langs,
        preset=preset,
    )

    # Run Benchmark
    deployment = DeploymentFactory.get_deployment(
        deployment_type=deployment_type,
        deployment_name=deployment_name,
        endpoint_url=endpoint_url,
        connections_name=connections_name,
    )
    evaluation = MTEB(
        tasks=validated_tasks,
        task_types=validated_task_types,
        task_langs=validated_task_langs,
        batch_size=batch_size,
    )
    logger.info("==========Benchmark started.==========")
    evaluation.run(model=deployment, verbosity=2, output_folder=output_metrics_dir)
    logger.info("==========Benchmark completed.==========")

    # Log params
    log_mlflow_params(
        tasks=",".join(validated_tasks) if validated_tasks and not preset else None,
        task_types=",".join(validated_task_types) if validated_task_types and not preset else None,
        task_langs=",".join(validated_task_langs) if validated_task_langs and not preset else None,
        preset=preset,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        endpoint_url=args.endpoint_url,
        deployment_type=args.deployment_type,
        deployment_name=args.deployment_name,
        connections_name=args.connections_name,
        tasks=args.tasks,
        task_types=args.task_types,
        task_langs=args.task_langs,
        preset=args.preset,
        batch_size=args.batch_size,
        output_metrics_dir=args.output_metrics_dir,
    )
