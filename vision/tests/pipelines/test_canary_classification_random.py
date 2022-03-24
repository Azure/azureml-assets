"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch
import argparse

from pipelines.canary.classification_random import build, build_arguments

# IMPORTANT: see conftest.py for fixtures

def test_canary_classification_random_pipeline_script(ml_client_instance_mock):
    """We're just testing if the code works on a basic level"""
    # create test arguments for the script
    script_args = [
        "--cpu_cluster", "foo",
        "--gpu_cluster", "bar",
        "--instance_count", "1",
        "--process_count_per_instance", "1",
    ]
    parser = argparse.ArgumentParser()
    parser = build_arguments(parser)
    args = parser.parse_args(script_args)

    pipeline_instance = build(ml_client_instance_mock, args)

    assert pipeline_instance.__class__.__name__ == "Pipeline"
