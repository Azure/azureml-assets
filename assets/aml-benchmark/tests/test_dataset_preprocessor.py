# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------
"""Test script for Dataset Preprocessor Component."""

import sys
import pytest
import json
import os
import glob
from typing import List, Dict, Any

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

from test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    get_src_dir,
    download_outputs,
    assert_logged_params,
    run_command
)


INPUT_DATASET = os.path.join(os.path.dirname(__file__), 'data/process_sample_examples.jsonl')
CUSTOM_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '../scripts/custom_dataset_preprocessors')


def _verify_and_get_output_records(
    inputs: List[str],
    outputs: str
) -> List[Dict[str, Any]]:
    """Verify the output and get output records.

    :param inputs: The list of input file with absolute path.
    :param outputs: Either path to output file or output directory containing output files.
    :return: list of json records
    """
    if not os.path.isfile(outputs):
        output_files = glob.glob(outputs + '/**/*.jsonl', recursive=True)
    else:
        output_files = [outputs]
    assert len(output_files) == 1
    with open(output_files[0], "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)
    for input in inputs:
        with open(input, "r") as f:
            input_records = [json.loads(line) for line in f]
            input_row_count = len(input_records)
            assert input_row_count == output_row_count
    return output_records


class TestDatasetPreprocessorComponent:
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
        exp_name = f"{self.EXP_NAME}"
        pipeline_job = self._get_pipeline_job(
            dataset,
            template_input,
            script_path,
            encoder_config,
            f"{self.test_dataset_preprocessor_as_component.__name__}-{dataset_name}",
            pipeline_file="dataset_preprocessor_pipeline.yaml",
        )
        pipeline_job = ml_client.create_or_update(pipeline_job, experiment_name=exp_name)
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)
        self._verify_and_get_output_records(
            pipeline_job, [dataset], output_dir=os.path.dirname(INPUT_DATASET)
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
        input_files: List[str],
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
            output_dir = Constants.OUTPUT_DIR.format(os.getcwd(), output_name=output_name)
        else:
            output_dir = Constants.OUTPUT_DIR.format(output_dir=output_dir, output_name=output_name)
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        records = _verify_and_get_output_records(input_files, output_dir)
        return records


class TestDatasetPreprocessorScript:
    """Testing the script."""

    @pytest.mark.parametrize(
        "dataset_name, dataset,template_input,script_path, encoder_config",
        [
            (
                "gsm8k", INPUT_DATASET,
                '{"question":{{question}}, "solution":{{answer.split("#### ")[0]}},"answer":{{answer.split("#### ")[-1]|string}}}',
                None, None
            ),
            ("quac", INPUT_DATASET, None, os.path.join(CUSTOM_SCRIPT_PATH, "quac_textgen_babel.py"), None),
        ],
    )
    def test_dataset_preprocessor_as_script(
        self,
        dataset_name: str,
        dataset: str,
        template_input: str,
        script_path: str,
        encoder_config: str,
        output_dataset: str = os.path.join(os.path.dirname(INPUT_DATASET), "processed_output.jsonl"),
    ) -> None:
        """Dataset Preprocessor script test."""
        src_dir = get_src_dir()
        with open(
            os.path.join(
                os.path.dirname(INPUT_DATASET), "process_one_example.jsonl"
                    ), "w"
                ) as writer:
            with open(INPUT_DATASET, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        writer.write(json.dumps(out_row) + "\n")
        dataset = os.path.join(os.path.dirname(INPUT_DATASET), "process_one_example.jsonl")
        argss = ["--dataset", dataset, "--output_dataset", output_dataset,]
        if template_input is not None:
            argss.extend(["--template_input", f"'{template_input}'"])
        elif script_path is not None:
            argss.extend(["--script_path", script_path])
        if encoder_config is not None:
            argss.extend(["--encoder_config", str(encoder_config)])
        argss = " ".join(argss)
        cmd = f"cd {src_dir} && python -m dataset_preprocessor.main {argss}"
        run_command(f"{cmd}")
        _verify_and_get_output_records([dataset], output_dataset)
        return

    def _verify_and_get_output_records(
        self,
        input_files: List[str],
        output_dataset: str
    ) -> List[Dict[str, Any]]:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param input_files: The list of input file with absolute path.
        :param output_dataset: path to output file.
        :rtype: List[Dict[str, Any]]
        """
        records = _verify_and_get_output_records(input_files, output_dataset)
        return records

    @pytest.mark.parametrize(
        "dataset_name, dataset, input_template, return_template",
        [
            (
                "mnli", INPUT_DATASET,
                """ {"premise":{{premise}}, "hypothesis":{{hypothesis}},"label":{{label}}} """,
                """ {"premise":{{premise|tojson}}, "hypothesis":{{hypothesis|tojson}},"label":{{label|tojson}}} """
            ),
        ],
    )
    def test_add_json_filter(
        self,
        dataset_name: str,
        dataset: str,
        input_template: str,
        return_template: str
    ):
        """Test if tojson filter is added to each value in the template."""
        sys.path.append(get_src_dir())
        from dataset_preprocessor import dataset_preprocessor as dsp
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
        obj = dsp.DatasetPreprocessor(input_dataset=dataset, template=input_template)
        template = obj.add_json_filter(obj.template)
        assert (template == return_template)
