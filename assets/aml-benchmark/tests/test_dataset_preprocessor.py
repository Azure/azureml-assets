# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------
"""Test script for Dataset Preprocessor Component."""

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from typing import List, Dict, Any
import pytest
import json
import os
from azure.ai.ml.constants import AssetTypes
from utils import (
    load_yaml_pipeline,
    get_mlclient,
    download_outputs,
    assert_logged_params,
)

INPUT_DATASET = os.path.join(os.getcwd(), 'data/process_sample_examples.jsonl')
CUSTOM_SCRIPT_PATH = os.path.join(os.getcwd(), '../scripts/custom_dataset_preprocessors')


class TestDatasetPreprocessor:
    """Testing the component."""

    EXP_NAME = "preprocessor-test"

    @pytest.mark.parametrize(
        "dataset_name, dataset,template_input,script_path, encoder_config",
        [
            (
                "gsm8k", INPUT_DATASET,
                """ {
                    "question":{{question}},
                    "solution":{{answer.split("#### ")[0]}},
                    "answer":{{answer.split("#### ")[-1]|string}}
                } """,
                None, None
            ),
            ("quac_org", INPUT_DATASET, None, os.path.join(CUSTOM_SCRIPT_PATH, "quac_textgen_babel.py"), None),
            (
                "mnli", INPUT_DATASET,
                """ {"premise":{{premise}}, "hypothesis":{{hypothesis}},"label":{{label|string}}} """,
                None, '{"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}'
            ),
        ],
    )
    def test_dataset_preprocessor_as_component(
        self,
        dataset_name: str,
        dataset: str,
        template_input: str,
        script_path: str,
        encoder_config: str,
    ) -> None:
        """Dataset Preprocessor component test."""
        with open(
            os.path.join(os.path.dirname(INPUT_DATASET), "process_one_example.jsonl"), "w"
        ) as writer:
            with open(INPUT_DATASET, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        dataset = os.path.join(os.path.dirname(INPUT_DATASET), "process_one_example.jsonl")
        ml_client = get_mlclient()
        exp_name = f"{self.EXP_NAME}-{dataset_name}"
        pipeline_job = self._get_pipeline_job(
            dataset,
            template_input,
            script_path,
            encoder_config,
            "Dataset pre-processor pipeline test",
            pipeline_file="dataset_preprocessor_pipeline.yaml",
        )
        # submit the pipeline job
        try:
            pipeline_job = ml_client.create_or_update(
                pipeline_job, experiment_name=exp_name
            )
            ml_client.jobs.stream(pipeline_job.name)
            print(pipeline_job)
        except Exception as e:
            print(e)
            print('Failed with exception')
        self._verify_and_get_output_records(
            pipeline_job
        )
        assert_logged_params(
            pipeline_job.name,
            exp_name,
            dataset=[dataset],
            template_input=template_input,
            script_path=script_path,
            encoder_config=encoder_config
        )

    def _get_pipeline_job(
        self,
        dataset: str,
        template_input: str,
        script_path: str,
        encoder_config: str,
        display_name: str,
        pipeline_file: str
    ) -> Job:
        pipeline_job = load_yaml_pipeline(pipeline_file)
        # set the pipeline inputs
        pipeline_job.inputs.dataset = Input(type=AssetTypes.URI_FILE, path=dataset)
        if template_input:
            pipeline_job.inputs.template_input = template_input
        else:
            pipeline_job.inputs.template_input = None
        if script_path:
            pipeline_job.inputs.script_path = Input(type=AssetTypes.URI_FILE, path=script_path)
        else:
            pipeline_job.inputs.script_path = None
        if encoder_config:
            pipeline_job.inputs.encoder_config = encoder_config
        else:
            pipeline_job.inputs.encoder_config = None
        pipeline_job.display_name = display_name
        return pipeline_job

    def _verify_and_get_output_records(
        self,
        job: Job,
        output_dir: str = None
    ) -> List[Dict[str, Any]]:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param output_dir: The local output directory to download pipeline outputs
        :type output_dir: str
        :return: output records
        :rtype: List[Dict[str, Any]]
        """
        output_name = job.outputs.output_dataset.port_name
        if not output_dir:
            output_dir = os.getcwd()
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
