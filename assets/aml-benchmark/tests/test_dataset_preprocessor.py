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
import subprocess

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

from .test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    get_src_dir,
    download_outputs,
    assert_logged_params,
    run_command,
    assert_exception_mssg
)


def _verify_and_get_output_records(
    dataset_name: str,
    input_file: str,
    expected_output_file: str,
    outputs: str
) -> None:
    """Verify the output and get output records.

    :param dataset_name: The path to jsonl file passed as input_dataset in pipeline.
    :param input_file: The path to jsonl file passed as input_dataset in pipeline or scripts.
    :param expected_output_file: The path to jsonl file containing expected outputs.
    :param outputs: Either path to output file or output directory containing output files.
    """
    with open(input_file, "r") as f:
        input_records = [json.loads(line) for line in f]
        input_row_count = len(input_records)
    expected_output_records = []
    with open(expected_output_file, "r") as f:
        for line in f:
            out = json.loads(line)
            if out.get('name') == dataset_name:
                del out['name']
                expected_output_records.append(out)
    expected_output_row_count = len(expected_output_records)
    if not os.path.isfile(outputs):
        output_files = glob.glob(outputs + '/**/*.jsonl', recursive=True)
    else:
        output_files = [outputs]
    assert len(output_files) == 1
    with open(output_files[0], "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)
    assert input_row_count == output_row_count == expected_output_row_count
    assert output_records == expected_output_records
    return


class TestDatasetPreprocessorComponent:
    """Testing the component."""

    EXP_NAME = "preprocessor-test"

    @pytest.mark.parametrize(
        "dataset_name, dataset,template_input,script_path, encoder_config",
        [
            (
                "gsm8k", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
                """ {
                    "question":{{question}},
                    "solution":{{answer.split("#### ")[0]}},
                    "answer":{{answer.split("#### ")[-1]|string}}
                } """,
                None, None,
            ),
            (
                "quac_org", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, None,
                os.path.join(Constants.CUSTOM_PREPROCESSOR_SCRIPT_PATH, "quac_textgen_babel.py"), None
            ),
            (
                "mnli", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
                """ {"premise":{{premise}}, "hypothesis":{{hypothesis}},"label":{{label|string}}} """,
                None, '{"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}'
            ),
            (
                "hellaswag_hf", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, None,
                os.path.join(Constants.CUSTOM_PREPROCESSOR_SCRIPT_PATH, "hellaswag_hf.py"), None
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
            os.path.join(
                os.path.dirname(
                    Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"), "w"
                ) as writer:
            with open(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_example.jsonl"
        )
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
        self._verify_and_get_output_records(
            pipeline_job, dataset_name, dataset,
            Constants.PREPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE,
            output_dir=os.path.join(os.path.dirname(__file__), 'data')
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
        dataset_name: str,
        input_file: str,
        expected_output_file: str,
        output_dir: str = None
    ) -> None:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param dataset_name: The path to jsonl file passed as input_dataset in pipeline.
        :type dataset_name: str
        :param input_file: The path to jsonl file passed as input_dataset in pipeline.
        :type input_file: str
        :param expected_output_file: The path to josnl file containing expected outputs for given inputs.
        :type expected_output_file
        :param output_dir: The local output directory to download pipeline outputs.
        :type output_dir: str
        """
        output_name = job.outputs.output_dataset.port_name
        if not output_dir:
            output_dir = Constants.OUTPUT_DIR.format(
                os.getcwd(), output_name=output_name
            )
        else:
            output_dir = Constants.OUTPUT_DIR.format(
                output_dir=output_dir, output_name=output_name
            )
        download_outputs(
            job_name=job.name, output_name=output_name,
            download_path=output_dir
        )
        _verify_and_get_output_records(
            dataset_name, input_file, expected_output_file, output_dir
        )
        return


class TestDatasetPreprocessorScript:
    """Testing the script."""

    @pytest.mark.parametrize(
        "dataset_name, dataset,template_input,script_path, encoder_config",
        [
            (
                "gsm8k", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
                """ {
                    "question":{{question}},
                    "solution":{{answer.split("#### ")[0]}},
                    "answer":{{answer.split("#### ")[-1]|string}}
                } """,
                None, None
            ),
            (
                "quac_org", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, None,
                os.path.join(Constants.CUSTOM_PREPROCESSOR_SCRIPT_PATH, "quac_textgen_babel.py"), None
            ),
            (
                "hellaswag_hf", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, None,
                os.path.join(Constants.CUSTOM_PREPROCESSOR_SCRIPT_PATH, "hellaswag_hf.py"), None
            )
        ],
    )
    def test_dataset_preprocessor_as_script(
        self,
        dataset_name: str,
        dataset: str,
        template_input: str,
        script_path: str,
        encoder_config: str,
        output_dataset: str = os.path.join(
            os.path.dirname(__file__), 'data/processed_output.jsonl'
        ),
    ) -> None:
        """Dataset Preprocessor script test."""
        src_dir = get_src_dir()
        with open(
            os.path.join(
                os.path.dirname(
                    Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"), "w"
                ) as writer:
            with open(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_example.jsonl"
        )
        argss = ["--dataset", dataset, "--output_dataset", output_dataset,]
        if dataset is not None:
            argss.extend(["--dataset", dataset])
        if template_input is not None:
            argss.extend(["--template_input", f"'{template_input}'"])
        elif script_path is not None:
            argss.extend(["--script_path", script_path])
        if encoder_config is not None:
            argss.extend(["--encoder_config", str(encoder_config)])
        argss = " ".join(argss)
        cmd = f"cd {src_dir} && python -m aml_benchmark.dataset_preprocessor.main {argss}"
        run_command(f"{cmd}")
        _verify_and_get_output_records(
            dataset_name, dataset,
            Constants.PREPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE,
            output_dataset
        )
        return

    @pytest.mark.parametrize(
        "dataset_name, dataset, input_template, return_template",
        [
            (
                "mnli", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
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
            os.path.join(os.path.dirname(
                Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"), "w"
        ) as writer:
            with open(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"
        )
        obj = dsp.DatasetPreprocessor(input_dataset=dataset, template=input_template)
        template = obj.add_json_filter(obj.template)
        assert (template == return_template)

    @pytest.mark.parametrize(
        "dataset_name, dataset,template_input,script_path, encoder_config",
        [
            (
                "gsm8k", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
                """ {
                    "question":{{question}},
                    "solution":{{answer.split("#### ")[0]}},
                    "answer":{{answer.split("#### ")[-1]|string}}
                } """,
                None, None
            )
        ],
    )
    def test_invalid_inputs(
        self,
        dataset_name: str,
        dataset: str,
        template_input: str,
        script_path: str,
        encoder_config: str
    ):
        """Test the exceptions raise during inputs validation."""
        invalid_dataset_error_mssg = (
            "the following arguments are required: --dataset"
        )
        invalid_jsonl_dataset_mssg = (
            "No .jsonl files found in the given input dataset."
        )
        invalid_preprocessor_logic_exception_mssg = (
           "Please provide the input to apply preprocessing logic either via template input or script_path."
        )
        invalid_user_script_mssg = (
            "Please provide python script containing your custom preprocessor logic."
        )
        src_dir = get_src_dir()
        try:
            argss = " ".join(["--template_input", f"'{template_input}'"])
            cmd = f"cd {src_dir} && python -m aml_benchmark.dataset_preprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            out_message = e.output.strip()
            assert invalid_dataset_error_mssg in out_message

        dummy_dataset_path = os.path.join(os.getcwd(), "input_dataset_path")
        os.system(f"mkdir {dummy_dataset_path}")
        try:
            argss = " ".join(["--dataset", dummy_dataset_path])
            cmd = f"cd {src_dir} && python -m aml_benchmark.dataset_preprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_jsonl_dataset_mssg)

        try:
            argss = " ".join(["--dataset", dataset])
            cmd = f"cd {src_dir} && python -m aml_benchmark.dataset_preprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_preprocessor_logic_exception_mssg)

        dummy_script_path = os.path.join(os.getcwd(), "user_script.json")
        os.system(f"touch {dummy_script_path}")
        try:
            argss = " ".join(["--dataset", dataset, "--script_path", dummy_script_path])
            cmd = f"cd {src_dir} && python -m aml_benchmark.dataset_preprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_user_script_mssg)

    @pytest.mark.parametrize(
        "dataset_name, dataset,template_input,script_path, encoder_config",
        [
            (
                "truthful_qa:generation", Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, None,
                os.path.join(Constants.CUSTOM_PREPROCESSOR_SCRIPT_PATH, "truthfulqa_hf.py"), None
            )
        ],
    )
    def test_truthfulqa_hf_dataset_preprocessor(
        self,
        dataset_name: str,
        dataset: str,
        template_input: str,
        script_path: str,
        encoder_config: str,
        output_dataset: str = os.path.join(
            os.path.dirname(__file__), 'data/processed_output.jsonl'
        ),
    ) -> None:
        """TruthfulQA-HF Dataset Custom Preprocessor script test."""
        src_dir = get_src_dir()
        with open(
            os.path.join(
                os.path.dirname(
                    Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"), "w"
                ) as writer:
            with open(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_example.jsonl"
        )
        argss = ["--dataset", dataset, "--output_dataset", output_dataset,]
        if dataset is not None:
            argss.extend(["--dataset", dataset])
        if template_input is not None:
            argss.extend(["--template_input", f"'{template_input}'"])
        elif script_path is not None:
            argss.extend(["--script_path", script_path])
        if encoder_config is not None:
            argss.extend(["--encoder_config", str(encoder_config)])
        argss = " ".join(argss)
        cmd = f"cd {src_dir} && python -m aml_benchmark.dataset_preprocessor.main {argss}"
        run_command(f"{cmd}")
        with open(dataset, "r") as f:
            input_records = [json.loads(line) for line in f]
        input_row_count = len(input_records)
        if not os.path.isfile(output_dataset):
            output_files = glob.glob(output_dataset + '/**/*.jsonl', recursive=True)
        else:
            output_files = [output_dataset]
        with open(output_files[0], "r") as f:
            output_records = [json.loads(line) for line in f]
        output_row_count = len(output_records)
        assert input_row_count == output_row_count
        expected_columns = [
            'question', 'best_answer', 'choices', 'best_answer_index',
            'correct_answers', 'labels', 'best_answer_label'
        ]
        assert list(output_records[0].keys()) == expected_columns
