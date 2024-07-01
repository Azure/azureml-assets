# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the Output Handler."""

import os

from abc import ABC, abstractmethod
from ..telemetry import logging_utils as lu


class OutputHandler(ABC):
    """An abstract class for handling output."""

    @abstractmethod
    def save_mini_batch_results(
            self,
            mini_batch_results: list,
            mini_batch_results_out_directory: str,
            raw_mini_batch_context):
        """Abstract save method."""
        pass


class SingleFileOutputHandler(OutputHandler):
    """Defines a class to emit all results to a single output file. This is used as the default output handler."""

    def save_mini_batch_results(
            self,
            mini_batch_results: list,
            mini_batch_results_out_directory: str,
            raw_mini_batch_context):
        """Save mini batch results to a single file."""
        lu.get_logger().debug("mini_batch_results_out_directory: {}".format(mini_batch_results_out_directory))

        filename = f"{raw_mini_batch_context.minibatch_index}.jsonl"
        file_path = os.path.join(mini_batch_results_out_directory, filename)

        lu.get_logger().debug(f"Start saving {len(mini_batch_results)} results to file {file_path}.")
        with open(file_path, "w", encoding="utf-8") as writer:
            for item in mini_batch_results:
                writer.write(item + "\n")

        lu.get_logger().info(f"Completed saving {len(mini_batch_results)} results to file {file_path}.")


class SeparateFileOutputHandler(OutputHandler):
    """Defines a class to emit successful results and errors to separate output files."""

    def save_mini_batch_results(
            self,
            mini_batch_results: list,
            mini_batch_results_out_directory: str,
            raw_mini_batch_context):
        """Save successful mini batch results and errors to two separate files."""
        lu.get_logger().debug("mini_batch_results_out_directory: {}".format(mini_batch_results_out_directory))

        os.makedirs(mini_batch_results_out_directory+"/results", exist_ok=True)
        success_filename = f"results/results_{raw_mini_batch_context.minibatch_index}.jsonl"
        success_file_path = os.path.join(mini_batch_results_out_directory, success_filename)

        os.makedirs(mini_batch_results_out_directory+"/errors", exist_ok=True)
        error_filename = f"errors/errors_{raw_mini_batch_context.minibatch_index}.jsonl"
        error_file_path = os.path.join(mini_batch_results_out_directory, error_filename)

        lu.get_logger().debug(f"Start saving {len(mini_batch_results)} results to files {success_file_path} \
                               and {error_file_path}.")
        with open(success_file_path, "w", encoding="utf-8") as success_writer, \
             open(error_file_path, "w", encoding="utf-8") as error_writer:
            for item in mini_batch_results:
                if '"status": "SUCCESS"' in item:
                    success_writer.write(item + "\n")
                else:
                    error_writer.write(item + "\n")

        lu.get_logger().info(f"Completed saving {len(mini_batch_results)} results to files {success_file_path} \
                               and {error_file_path}.")
