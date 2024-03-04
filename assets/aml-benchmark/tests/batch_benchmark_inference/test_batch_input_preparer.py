# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Batch inference preparer Component."""

from typing import Optional
import json
import os
import uuid
import pytest

from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from ..test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    ModelType,
    PromptType,
)


class TestBatchInferencePreparerComponent:
    """Component test for batch inference preparer."""

    EXP_NAME = "batch-inference-preparer-test"
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

    @pytest.mark.parametrize('model_type, prompt_type', [
        (ModelType.OAI, PromptType.CHAT_COMPLETION),
        (ModelType.OAI, PromptType.TEXT_COMPLETION),
        (ModelType.OSS, PromptType.CHAT_COMPLETION),
        (ModelType.OSS, PromptType.TEXT_COMPLETION),
        (ModelType.VISION_OSS, None)
    ])
    def test_batch_inference_preparer(self, model_type: ModelType, prompt_type: Optional[PromptType], temp_dir: str):
        """Test batch inference preparer."""
        ml_client = get_mlclient()
        score_url = "https://test.com"
        param_dict = {}

        # llm related params
        temperature = 0.001
        top_p = 0.5
        max_new_tokens = 50
        ignore_eos = True
        return_full_text = False
        prompt_col = "prompt"

        if model_type == ModelType.OAI:
            param_dict = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens
            }
            if prompt_type == PromptType.CHAT_COMPLETION:
                key = "messages"
                pattern = json.dumps({
                    key: [{"role": "system", "content": ""}, {"role": "user", "content": f"###<{prompt_col}>"}],
                    **param_dict
                })
            elif prompt_type == PromptType.TEXT_COMPLETION:
                key = "prompt"
                pattern = json.dumps({
                    key: f"###<{prompt_col}>",
                    **param_dict
                })
        elif model_type == ModelType.OSS:
            param_dict = {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": ignore_eos
            }
            key = "input_data"
            if prompt_type == PromptType.CHAT_COMPLETION:
                pattern = json.dumps({
                    key: {
                        "input_string": [{"role": "user", "content": f"###<{prompt_col}>"}],
                        "parameters": param_dict
                    }
                })
            elif prompt_type == PromptType.TEXT_COMPLETION:
                param_dict["return_full_text"] = return_full_text
                pattern = json.dumps({
                    key: {
                        "input_string": [ f"###<{prompt_col}>" ],
                        "parameters": param_dict
                    }
                })
        elif model_type == ModelType.VISION_OSS:
            pattern = self.VISION_REQUEST
            key = "input_data"

        pipeline_job = self._get_pipeline_job(
            self.test_batch_inference_preparer.__name__,
            pattern,
            endpoint_url=score_url,
            model_type=model_type,
            temp_dir=temp_dir,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        self._verify_output(
            pipeline_job, output_dir=out_dir, check_key=key,
            model_type=model_type,
            check_param_dict=param_dict
        )

    def _get_pipeline_job(
                self,
                display_name: str,
                batch_input_pattern: str,
                endpoint_url: str,
                model_type: ModelType,
                temp_dir: Optional[str] = None,
            ) -> Job:
        pipeline_job = load_yaml_pipeline("batch_inference_preparer.yaml")

        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".jsonl")
            batch_input_file = Constants.BATCH_INFERENCE_PREPARER_FILE_PATH_VISION \
                if model_type == ModelType.VISION_OSS else Constants.BATCH_INFERENCE_PREPARER_FILE_PATH
            with open(batch_input_file, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())

        # set the pipeline inputs
        if model_type == ModelType.VISION_OSS:
            pipeline_job.jobs['run_batch_inference_preparer'].inputs.model_type = model_type.value
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
            assert check_key in r, f"{check_key} not in records {r}"
            if model_type == ModelType.VISION_OSS:
                assert 'data' in r[check_key]
            elif model_type == ModelType.OSS:
                for k, v in check_param_dict.items():
                    assert r[check_key]['parameters'][k] == v, f"{k} not equal to {v}"
            elif model_type == ModelType.OAI:
                for k, v in check_param_dict.items():
                    assert r[k] == v, f"{k} not equal to {v}"
