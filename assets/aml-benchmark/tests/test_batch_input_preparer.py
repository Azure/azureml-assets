# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Batch inference preparer Component."""

from typing import Optional
import json
import os
import uuid
import pytest
import sys
import tempfile
import shutil

from .test_utils import get_src_dir

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
sys.path.append(get_src_dir())
print(get_src_dir())

from aml_benchmark.batch_inference_preparer.main import main as preparer_main  # noqa: E402

from .test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs
)


@pytest.fixture
def input_dataset(tmp_path):
    """Create input dataset."""
    target_output = os.path.join(tmp_path, 'input_dataset')
    os.makedirs(target_output, exist_ok=True)
    shutil.copy(
        Constants.BATCH_INFERENCE_PREPARER_FILE_PATH_VISION,
        os.path.join(target_output, 'input.jsonl'))
    return target_output


class TestBatchInferencePreparerComponent:
    """Component test for batch inference preparer."""

    EXP_NAME = "batch-inference-preparer-test"
    LLM_REQUEST = ('{'
                   '   "input_data":'
                   '   {'
                   '       "input_string": ["###<prompt>"],'
                   '       "parameters":'
                   '       {'
                   '           "temperature": 0.6,'
                   '           "max_new_tokens": 100,'
                   '           "do_sample": true'
                   '       }'
                   '   },'
                   '   "_batch_request_metadata": ###<_batch_request_metadata>'
                   '}')
    VISION_REQUEST = ('{'
                      '"input_data": {'
                      '  "columns": ['
                      '    "image",'
                      '    "text"'
                      '  ],'
                      '  "index": [0],'
                      '  "data": ['
                      '    ["###<image>", "###<text>"]'
                      '  ]'
                      '},'
                      '"params": {}'
                      '}')

    @pytest.mark.parametrize(
            'model_type', [(None), ("vision_oss")]
    )
    def test_batch_inference_preparer(self, model_type: str, temp_dir: str):
        """Test batch inference preparer."""
        ml_client = get_mlclient()
        score_url = "https://test.com"
        request = self.VISION_REQUEST if model_type == "vision_oss" else self.LLM_REQUEST
        pipeline_job = self._get_pipeline_job(
            self.test_batch_inference_preparer.__name__,
            request,
            endpoint_url=score_url,
            temp_dir=temp_dir,
            model_type=model_type,
        )
        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        if model_type == "vision_oss":
            expected_keys = ["input_data"]
        else:
            expected_keys = ["input_data", "_batch_request_metadata"]
        self._verify_output(
            pipeline_job, output_dir=out_dir, check_key=expected_keys,
            model_type=model_type,
            check_param_dict={"temperature": 0.6, "max_new_tokens": 100, "do_sample": True}
        )

    def _get_pipeline_job(
                self,
                display_name: str,
                batch_input_pattern: str,
                endpoint_url: str,
                temp_dir: Optional[str] = None,
                model_type: Optional[str] = None,
            ) -> Job:
        pipeline_job = load_yaml_pipeline("batch_inference_preparer.yaml")

        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".jsonl")
            batch_input_file = Constants.BATCH_INFERENCE_PREPARER_FILE_PATH_VISION \
                if model_type == "vision_oss" else Constants.BATCH_INFERENCE_PREPARER_FILE_PATH
            with open(batch_input_file, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())

        # set the pipeline inputs
        if model_type:
            pipeline_job.jobs['run_batch_inference_preparer'].inputs.model_type = model_type
        pipeline_job.inputs.input_dataset = Input(
            type="uri_folder", path=temp_dir
        )
        pipeline_job.inputs.batch_input_pattern = batch_input_pattern
        pipeline_job.inputs.endpoint_url = endpoint_url
        pipeline_job.display_name = display_name
        pipeline_job.name = str(uuid.uuid4())

        return pipeline_job

    def _verify_output(self, job, output_dir, check_key, model_type, check_param_dict):
        output_name = job.outputs.formatted_data.port_name
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        output_dir = os.path.join(output_dir, "named-outputs")
        print(output_dir)
        output_file_path = os.path.join(output_dir, "formatted_data", "formatted_data.json")
        # Read the output file
        with open(output_file_path, "r") as f:
            output_records = [json.loads(line) for line in f]
        for r in output_records:
            for k in check_key:
                assert k in r, f"{k} not in records {r}"
            if model_type == "oss" or model_type is None:
                for k, v in check_param_dict.items():
                    assert r['input_data']['parameters'][k] == v, f"{k} not equal to {v}"
            if model_type == "vision_oss":
                assert 'data' in r['input_data']


class TestInferencePreparer:
    """Test InferencePreparer."""

    def test_batch_inference_preparer_main(self, input_dataset):
        output_path = tempfile.TemporaryDirectory().name
        formatted_dataset = os.path.join(output_path, "formatted_data")
        output_metadat = os.path.join(output_path, "output_metadata")
        os.makedirs(formatted_dataset, exist_ok=True)
        os.makedirs(output_metadat, exist_ok=True)
        preparer_main(
            input_dataset=input_dataset,
            formatted_dataset=formatted_dataset,
            model_type="oai",
            batch_input_pattern=None,
            n_samples=10,
            endpoint_url="an_endpoint",
            is_performance_test=False,
            label_key="label",
            deployment_config_dir=None,
            output_metadata=output_metadat,
            bypass_dataset_conversion=True
        )
