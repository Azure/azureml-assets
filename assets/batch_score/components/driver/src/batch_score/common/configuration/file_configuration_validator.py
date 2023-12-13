# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File configuration validator."""

import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


class InvalidConfigurationError(Exception):
    pass


_current_file_path = Path(os.path.abspath(__file__))
SCHEMAS_ROOT = _current_file_path.parent / "schemas"


class FileConfigurationValidator:
    def __init__(self, schema_file=None):
        if schema_file is None:
            schema_file = SCHEMAS_ROOT / "configuration.json"

        schema = _load_file(schema_file)
        registry = _get_registry()

        self._validator = Draft202012Validator(schema=schema, registry=registry)

    def validate(self, instance):
        instance = _load_file(instance)
        errors = list(self._validator.iter_errors(instance=instance))
        if errors:
            self._report_errors(errors)
            raise InvalidConfigurationError()

        instance = self._apply_defaults(instance)
        return instance

    def _report_errors(self, errors):
        print("Errors found while validating configuration.")
        for error in errors:
            context = [
                f"Context {i}: {self._get_error_message(c)}"
                for i, c in enumerate(error.context)
            ]
            message_parts = [self._get_error_message(error), *context]
            print("\n\n    ".join(message_parts))
            print("\n\n")

    def _get_error_message(self, error):
        return " -- ".join(
            [
                f"Error type: {error.validator}",
                f"Error message: {error.message}",
                f"JSON path: {error.json_path}",
                f"Instance: {error.instance}",
                f"Schema: {error.schema}",
            ]
        )

    def _apply_defaults(self, instance):
        # api - completion
        instance["api"].setdefault("response_segment_size", 0)

        # api - embedding
        instance["api"].setdefault("batch_size_per_request", 1)

        # authentication: no defaults

        # concurrency_settings
        instance.setdefault("concurrency_settings", {})
        instance["concurrency_settings"].setdefault("initial_worker_count", 100)
        instance["concurrency_settings"].setdefault("max_worker_count", 200)

        # inference_endpoint: no defaults

        # request_settings
        instance.setdefault("request_settings", {})
        instance["request_settings"].setdefault("headers", {})
        instance["request_settings"].setdefault("properties", {})
        instance["request_settings"].setdefault("timeout", 600)

        # log_settings
        instance.setdefault("log_settings", {})
        instance["log_settings"].setdefault("app_insights_log_level", "debug")
        instance["log_settings"].setdefault("stdout_log_level", "debug")

        # output_settings
        instance.setdefault("output_settings", {})
        instance["output_settings"].setdefault("ensure_ascii", False)
        instance["output_settings"].setdefault("save_partitioned_scoring_results", True)

        return instance


def list_files_recursively(dir):
    file_paths = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            file_paths.append(os.path.join(path, name))

    return file_paths


def _load_file(file):
    with open(file, "r") as f:
        return json.load(f)


def _get_registry():
    resources = [
        (
            referenced_schema_file,
            Resource.from_contents(_load_file(referenced_schema_file)),
        )
        for referenced_schema_file in list_files_recursively(SCHEMAS_ROOT)
    ]

    return Registry().with_resources(resources)
