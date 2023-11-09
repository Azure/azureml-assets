# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Test script for Inference Postprocessor Component."""

import pytest
import json
import os
import glob
import sys
import subprocess
from typing import List

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

from .test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    assert_logged_params,
    get_src_dir,
    run_command,
    assert_exception_mssg,
)


INPUT_PATH = os.path.join(os.path.dirname(__file__), "data")


def _verify_and_get_output_records(
    dataset_name: str, expected_output_file: str, outputs: str, input_file: str = None
) -> None:
    """Verify the output and get output records.

    :param dataset_name: The path to jsonl file passed as input_dataset in pipeline.
    :param input_file: The path to jsonl file passed as input_dataset in pipeline or scripts.
    :param expected_output_file: The path to jsonl file containing expected outputs.
    :param outputs: Either path to output file or output directory containing output files.
    """
    if input_file:
        with open(input_file, "r") as f:
            input_records = [json.loads(line) for line in f]
            input_row_count = len(input_records)
    expected_output_records = []
    with open(expected_output_file, "r") as f:
        for line in f:
            out = json.loads(line)
            if out.get("name") == dataset_name:
                del out["name"]
                expected_output_records.append(out)
    expected_output_row_count = len(expected_output_records)
    if not os.path.isfile(outputs):
        output_files = glob.glob(outputs + "/**/*.jsonl", recursive=True)
    else:
        output_files = [outputs]
    assert len(output_files) == 1
    with open(output_files[0], "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)
    if input_file:
        assert input_row_count == output_row_count == expected_output_row_count
    else:
        assert output_row_count == expected_output_row_count
    assert output_records == expected_output_records
    return


class TestInferencePostprocessorComponent:
    """Test Inference Postprocessor."""

    EXP_NAME = "postprocessor-test"

    @pytest.mark.parametrize(
        "dataset_name, prediction_dataset, prediction_column_name, ground_truth_dataset, ground_truth_column_name, \
        separator, regex_expr, remove_prefixes, strip_characters, extract_number, template, script_path, \
        label_map, find_first",
        [
            (
                "gsm8k", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", None,
                None, None, None, None, """{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}""",
                None, None, None,
            ),
            (
                "human-eval", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "samples",
                None, None, None, "^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)", None, None, None,
                None, None, None, None,
            ),
            (
                "gsm8k_multiple_preds", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", None, None,
                None, None, None, """{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}""",
                None, None, None,
            ),
            (
                "human-eval_multiple_preds", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "samples",
                None, None, None, "^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)", None, None, None, None, None,
                None, None,
            ),
        ],
    )
    def test_inference_postprocessor_as_component(
        self,
        dataset_name: str,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        remove_prefixes: str,
        strip_characters: str,
        extract_number: str,
        template: str,
        script_path: str,
        label_map: str,
        find_first: str,
    ) -> None:
        """Inference Postprocessor component test."""
        if ground_truth_dataset:
            with open(
                os.path.join(
                    os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                    "process_one_example.jsonl",
                ),
                "w",
            ) as writer:
                with open(
                    Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "r"
                ) as reader:
                    for line in reader:
                        out_row = json.loads(line)
                        if out_row.get("name") == dataset_name:
                            del out_row["name"]
                            writer.write(json.dumps(out_row) + "\n")
            ground_truth_dataset = os.path.join(
                os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                "process_one_example.jsonl",
            )
        with open(
            os.path.join(
                os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                "process_one_prediction_example.jsonl",
            ),
            "w",
        ) as writer:
            with open(
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "r"
            ) as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get("name") == dataset_name:
                        del out_row["name"]
                        writer.write(json.dumps(out_row) + "\n")
        prediction_dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_prediction_example.jsonl",
        )
        ml_client = get_mlclient()
        exp_name = f"{self.EXP_NAME}"
        pipeline_job = self._get_pipeline_job(
            prediction_dataset,
            prediction_column_name,
            ground_truth_dataset,
            ground_truth_column_name,
            separator,
            regex_expr,
            remove_prefixes,
            strip_characters,
            extract_number,
            template,
            script_path,
            label_map,
            find_first,
            self.test_inference_postprocessor_as_component.__name__,
            pipeline_file="inference_postprocessor_pipeline.yaml",
        )
        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        self._verify_and_get_output_records(
            pipeline_job,
            dataset_name,
            ground_truth_dataset,
            Constants.POSTPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE,
            output_dir=os.path.join(os.path.dirname(__file__), "data"),
        )
        assert_logged_params(
            pipeline_job.name,
            exp_name,
            prediction_dataset=[prediction_dataset],
            prediction_column_name=prediction_column_name,
            ground_truth_dataset=[ground_truth_dataset]
            if ground_truth_dataset
            else None,
            ground_truth_column_name=ground_truth_column_name,
            separator=separator,
            regex_expr=regex_expr,
            remove_prefixes=remove_prefixes,
            strip_characters=strip_characters,
            extract_number=extract_number,
            template=template,
            script_path=script_path,
            label_map=label_map,
            find_first=find_first
        )

    def _get_pipeline_job(
        self,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        remove_prefixes: str,
        strip_characters: str,
        extract_number: str,
        template: str,
        script_path: str,
        label_map: str,
        find_first: str,
        display_name: str,
        pipeline_file: str,
    ) -> Job:
        """Get the pipeline job."""
        pipeline_job = load_yaml_pipeline(pipeline_file)
        # set the pipeline inputs
        pipeline_job.inputs.prediction_dataset = Input(
            type=AssetTypes.URI_FILE, path=prediction_dataset
        )
        pipeline_job.inputs.prediction_column_name = prediction_column_name
        if ground_truth_dataset:
            pipeline_job.inputs.ground_truth_dataset = Input(
                type=AssetTypes.URI_FILE, path=ground_truth_dataset
            )
        else:
            pipeline_job.inputs.ground_truth_dataset = None
        pipeline_job.inputs.ground_truth_column_name = ground_truth_column_name if ground_truth_column_name else None
        pipeline_job.inputs.separator = separator if separator else None
        pipeline_job.inputs.regex_expr = regex_expr if regex_expr else None
        pipeline_job.inputs.remove_prefixes = remove_prefixes if remove_prefixes else None
        pipeline_job.inputs.strip_characters = strip_characters if strip_characters else None
        pipeline_job.inputs.extract_number = extract_number if extract_number else None
        pipeline_job.inputs.template = template if template else None
        if script_path:
            pipeline_job.inputs.script_path = Input(
                type=AssetTypes.URI_FILE, path=script_path
            )
        else:
            pipeline_job.inputs.script_path = None
        pipeline_job.inputs.label_map = label_map if label_map else None
        pipeline_job.inputs.find_first = find_first if find_first else None
        pipeline_job.display_name = display_name
        pipeline_job.description = "Post processor component testing"
        return pipeline_job

    def _verify_and_get_output_records(
        self,
        job: Job,
        dataset_name: str = None,
        input_file: str = None,
        expected_output_file: str = None,
        output_dir: str = None,
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
        output_name = job.outputs.output_dataset_result.port_name
        if not output_dir:
            output_dir = Constants.OUTPUT_DIR.format(
                os.getcwd(), output_name=output_name
            )
        else:
            output_dir = Constants.OUTPUT_DIR.format(
                output_dir=output_dir, output_name=output_name
            )
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        _verify_and_get_output_records(
            dataset_name, expected_output_file, output_dir, input_file=input_file
        )
        return


class TestInferencePostprocessorScript:
    """Testing the script."""

    @pytest.mark.parametrize(
        "dataset_name, prediction_dataset, prediction_column_name, ground_truth_dataset, ground_truth_column_name, \
        separator, regex_expr, remove_prefixes, strip_characters, extract_number, template, script_path, \
        label_map, find_first",
        [
            (
                "gsm8k", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", None,
                None, None, None, None, """{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}""",
                None, None, None,
            ),
            (
                "human-eval", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "samples",
                None, None, None, "^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)", None, None, None,
                None, None, None, None,
            ),
            (
                "gsm8k_multiple_preds", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", None, None,
                None, None, None, """{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}""",
                None, None, None,
            ),
            (
                "human-eval_multiple_preds", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "samples",
                None, None, None, "^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)", None, None, None, None, None,
                None, None,
            ),
            (
                "gsm8k", Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", "\n\n",
                None, None, ".", "last", None, None, None, None,
            ),
        ],
    )
    def test_inference_postprocessor_as_script(
        self,
        dataset_name: str,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        remove_prefixes: str,
        strip_characters: str,
        extract_number: str,
        template: str,
        script_path: str,
        label_map: str,
        find_first: str,
        output_dataset: str = os.path.join(INPUT_PATH, "postprocessed_output.jsonl"),
    ) -> None:
        """Inference Postprocessor script test."""
        if ground_truth_dataset:
            with open(
                os.path.join(
                    os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                    "process_one_example.jsonl",
                ),
                "w",
            ) as writer:
                with open(
                    Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "r"
                ) as reader:
                    for line in reader:
                        out_row = json.loads(line)
                        if out_row.get("name") == dataset_name:
                            del out_row["name"]
                            writer.write(json.dumps(out_row) + "\n")
            ground_truth_dataset = os.path.join(
                os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                "process_one_example.jsonl",
            )
        with open(
            os.path.join(
                os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                "process_one_prediction_example.jsonl",
            ),
            "w",
        ) as writer:
            with open(
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "r"
            ) as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get("name") == dataset_name:
                        del out_row["name"]
                        writer.write(json.dumps(out_row) + "\n")
        prediction_dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_prediction_example.jsonl",
        )
        argss = [
            "--prediction_dataset",
            prediction_dataset,
            "--prediction_column_name",
            prediction_column_name,
            "--output_dataset",
            output_dataset,
        ]
        if ground_truth_dataset is not None:
            argss.extend(["--ground_truth_dataset", ground_truth_dataset])
        if ground_truth_column_name is not None:
            argss.extend(["--ground_truth_column_name", ground_truth_column_name])
        if template is not None:
            argss.extend(["--template", f"'{template}'"])
        if script_path is not None:
            argss.extend(["--script_path", script_path])
        if separator is not None:
            argss.extend(["--separator", f"'{separator}'"])
        if regex_expr is not None:
            argss.extend(["--regex_expr", f"'{regex_expr}'"])
        if remove_prefixes is not None:
            argss.extend(["--remove_prefixes", f"'{remove_prefixes}'"])
        if strip_characters is not None:
            argss.extend(["--strip_characters", f"'{strip_characters}'"])
        if extract_number is not None:
            argss.extend(["--extract_number", f"'{extract_number}'"])
        if find_first is not None:
            argss.extend(["--find_first", f"'{find_first}'"])
        if label_map is not None:
            argss.extend(["--label_map", f"'{label_map}'"])
        argss = " ".join(argss)
        src_dir = get_src_dir()
        cmd = f"cd {src_dir} && python -m aml_benchmark.inference_postprocessor.main {argss}"
        run_command(f"{cmd}")
        _verify_and_get_output_records(
            dataset_name,
            Constants.POSTPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE,
            output_dataset,
            input_file=ground_truth_dataset,
        )
        return

    @pytest.mark.parametrize(
        "prediction_dataset, prediction_column_name, template",
        [
            (
                os.path.join(INPUT_PATH, "sample_predictions.jsonl"),
                "prediction",
                '{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}',
            ),
        ],
    )
    def test_invalid_inputs(
        self, prediction_dataset: str, prediction_column_name: str, template: str
    ):
        """Test the exceptions raise during inputs validation."""
        invalid_prediction_dataset_error_mssg = (
            "the following arguments are required: --prediction_dataset"
        )
        invalid_jsonl_dataset_mssg = (
            "No .jsonl files found in the given prediction dataset."
        )
        invalid_prediction_colname_error_mssg = (
            "the following arguments are required: --prediction_column_name"
        )
        invalid_user_script_mssg = (
            "Please provide python script containing your custom postprocessor logic."
        )
        src_dir = get_src_dir()
        try:
            argss = " ".join(
                [
                    "--prediction_column_name",
                    prediction_column_name,
                    "--template",
                    f"'{template}'",
                ]
            )
            cmd = f"cd {src_dir} && python -m aml_benchmark.inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            out_message = e.output.strip()
            assert invalid_prediction_dataset_error_mssg in out_message
        try:
            argss = " ".join(
                [
                    "--prediction_dataset",
                    prediction_dataset,
                    "--template",
                    f"'{template}'",
                ]
            )
            cmd = f"cd {src_dir} && python -m aml_benchmark.inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            out_message = e.output.strip()
            assert invalid_prediction_colname_error_mssg in out_message

        dummy_dataset_path = os.path.join(os.getcwd(), "prediction_dataset_path")
        os.system(f"mkdir {dummy_dataset_path}")
        try:
            argss = " ".join(
                [
                    "--prediction_dataset",
                    dummy_dataset_path,
                    "--prediction_column_name",
                    prediction_column_name,
                    "--template",
                    f"'{template}'",
                ]
            )
            cmd = f"cd {src_dir} && python -m aml_benchmark.inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_jsonl_dataset_mssg)

        dummy_script_path = os.path.join(os.getcwd(), "user_script.json")
        os.system(f"touch {dummy_script_path}")
        try:
            argss = " ".join(
                [
                    "--prediction_dataset",
                    prediction_dataset,
                    "--prediction_column_name",
                    prediction_column_name,
                    "--script_path",
                    dummy_script_path,
                ]
            )
            cmd = f"cd {src_dir} && python -m aml_benchmark.inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_user_script_mssg)

    @pytest.mark.parametrize(
        "find_first, mock_completion_list, expected_completion_list",
        [
            ("foo", ["foo"], ["foo"]),
            ("foo", ["foo_def"], ["foo"]),
            ("foo", ["abc_foo"], ["foo"]),
            ("foo", ["abc_foo_def"], ["foo"]),
            ("foo", ["abcdef"], [""]),
            ("foo", [""], [""]),
            ("foo,bar", ["bar_foo"], ["bar"]),
            ("bar,foo", ["bar_foo"], ["bar"]),
            ("foo,bar", ["abc_bar_foo", "foo_bar"], ["bar", "foo"]),
            ("", ["foo"], ["foo"]),
            (None, ["foo"], ["foo"]),
        ],
    )
    def test_apply_find_first(
        self,
        find_first: str,
        mock_completion_list: List,
        expected_completion_list: List,
    ) -> None:
        """Test apply_find_first."""
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            find_first=find_first,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_find_first(text=input))
        print(
            f"Input: {mock_completion_list}\nOutput: {output}\nExpected: {expected_completion_list}\n\n"
        )
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "separator, mock_completion_list, expected_completion_list",
        [
            ("|", ["foo|bar"], ["foo"]),
            ("|", ["foo"], ["foo"]),
            ("|", ["|foo"], [""]),
            ("|", ["|"], [""]),
            ("|", [""], [""]),
            ("|||", ["foo|||", "bar"], ["foo", "bar"]),
            (" ", ["foo bar"], ["foo"]),
            (None, ["foo"], ["foo"]),
            (None, ["foo|bar"], ["foo|bar"]),
            ("", ["foo"], ["foo"]),
            ("", ["foo|bar"], ["foo|bar"]),
        ],
    )
    def test_apply_separator(
        self, separator: str, mock_completion_list: List, expected_completion_list: List
    ) -> None:
        """Test apply_separator."""
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            separator=separator,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_separator(text=input))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "regex, mock_completion_list, expected_completion_list",
        [
            ("foo", ["foo"], ["foo"]),
            (" (.*)", ["foo"], ["foo"]),
            (" ([-+]?\\d+)[.]{0,1}$", ["abc 123"], ["123"]),
            (r" ([-+]?\d+)[.]{0,1}$", ["abc 123"], ["123"]),
            (r"""([-+]?\d+)[.]{0,1}$""", ["abc 123"], ["123"]),
            (" ([-+]?\\d+)[.]{0,1}$", ["abc 123."], ["123"]),
            (" ([-+]?\\d+)[.]{0,1}$", ["abc -123."], ["-123"]),
            (" ([-+]?\\d+)[.]{0,1}$", ["abc123.0"], ["abc123.0"]),
            ("answer is (Yes|No)", ["answer is Yes"], ["Yes"]),
            ("answer is (Yes|No)", ["answer is No"], ["No"]),
            (
                "answer is (Yes|No)",
                ["answer is Maybe"],
                ["answer is Maybe"],
            ),  # NOTE: this is not intiuitive!
            ('"Answer: ([^\\n]+?)(?:\\.|\\n|$)"', ['"Answer: ] ) )."'], ["] ) )"]),
            (
                "^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)",
                [
                    '   for i in range(len(numbers) - 1):\n        \
                        for j in range(i + 1, len(numbers)):\n            \
                        if abs(numbers[i] - numbers[j]) < threshold:\n                \
                        return True\n    return False\n\n\ndef \
                        has_close_elements_2(numbers: List[float], threshold: float) -> bool:\n    \
                        """ Check if in given list of numbers, are any two numbers closer to each other than\n    \
                        given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    \
                        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    \
                        for i in range(len(numbers) - 1):\n        \
                        for j in range(i + 1, len(numbers)):\n            \
                        if abs(numbers[i] - numbers[j]) < threshold:\n                \
                        return True\n    \
                        return False\n\n\ndef has_close_elements_3(numbers: List[float], threshold: float) \
                        -> bool:\n    \
                        """ Check if in given list of numbers, are any two numbers closer to each other than\n    \
                        given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    \
                        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \
                        """\n    for i in range(len(numbers) - 1):\n        \
                        for j in range(i + 1, len(numbers)):\n            \
                        if abs(numbers[i] - numbers[j]) < threshold:\n                \
                        return True\n    \
                        return False\n\n\ndef has_close_elements_4(numbers: List[float], threshold: float) \
                        -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each \
                        other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0],'
                ],
                [
                    "   for i in range(len(numbers) - 1):\n        \
                        for j in range(i + 1, len(numbers)):\n            \
                        if abs(numbers[i] - numbers[j]) < threshold:\n                \
                        return True\n    return False\n\n"
                ],
            ),
        ],
    )
    def test_apply_regex_expr(
        self, regex: str, mock_completion_list: List, expected_completion_list: List
    ) -> None:
        """Test regex_expr."""
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            regex_expr=regex,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_regex_expr(text=input))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "mock_completion_list, expected_completion_list",
        [
            (["foo123"], ["123"]),
            (["foo123", "456foo123"], ["123", "456"]),
            (["bar", "foo456", "", "foo"], ["0", "456", "0", "0"]),
            (["4foo3"], ["4"]),
            (["-4foo3"], ["-4"]),
            (["foo-123"], ["-123"]),
            (["foo -123"], ["-123"]),
            (["42"], ["42"]),
            (["42foo"], ["42"]),
            (["123.456foo7.8"], ["123.456"]),
            (["-123.456foo7.8"], ["-123.456"]),
            (["-123.456 foo7.8"], ["-123.456"]),
            (["-.456"], ["-.456"]),
            (["--.456"], ["-.456"]),
            (["+56"], ["56"]),
            ([",56"], ["56"]),
            (["56,"], ["56"]),
            (["5,6"], ["5"]),
            (["5, 6"], ["5"]),
            (["12,654"], ["12654"]),
            (["12 654"], ["12654"]),
            (["12 64"], ["12"]),
            ([",654"], ["654"]),
            (["654,"], ["654"]),
            ([" 654"], ["654"]),
            (["654 "], ["654"]),
            ([".."], ["0"]),
        ],
    )
    def test_apply_extract_number(
        self, mock_completion_list: List, expected_completion_list: List
    ) -> None:
        """Test when extract_number is set to 'first'."""
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            extract_number="first",
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_extract_number(text=input))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "mock_completion_list, expected_completion_list",
        [
            (["foo123"], ["123"]),
            (["foo123", "456foo123"], ["123", "123"]),
            (
                ["bar", "123foo456", "", "foo"],
                ["0", "456", "0", "0"],
            ),  # 0 is the default value
            (["4foo3"], ["3"]),
            (["42"], ["42"]),
            (["42foo"], ["42"]),
            (["123.456foo7.8"], ["7.8"]),
            (["123.456foo-7.8"], ["-7.8"]),
            (["--7.8"], ["-7.8"]),
            (["123.456foo-.8"], ["-.8"]),
            (["123.456foo -.8"], ["-.8"]),
            (["+56"], ["56"]),
            ([",56"], ["56"]),
            (["56,"], ["56"]),
            (["5,6"], ["6"]),
            (["5, 6"], ["6"]),
            (["12,654"], ["12654"]),
            (["12 654"], ["12654"]),
            (["12 64"], ["64"]),
            ([",654"], ["654"]),
            (["654,"], ["654"]),
            ([" 654"], ["654"]),
            (["654 "], ["654"]),
            ([".."], ["0"]),
        ],
    )
    def test_apply_extract_number_last(
        self,
        mock_completion_list: List,
        expected_completion_list: List,
    ) -> None:
        """Test when extract_number is set to 'last'."""
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj1 = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            extract_number="last",
        )
        output = []
        for input in mock_completion_list:
            output.append(obj1.apply_extract_number(text=input))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "prefixes, mock_completion_list, expected_completion_list",
        [
            ("<prefix>", ["<prefix>foo"], ["foo"]),
            ("<wrong_prefix>", ["<prefix>foo"], ["<prefix>foo"]),
            ("<prefix>", ["<prefix>foo", "<prefix>bar"], ["foo", "bar"]),
            ("", ["foo"], ["foo"]),
            ("<prefix>", [" <prefix>foo"], [" <prefix>foo"]),
            ("prefix1,prefix2", ["prefix1 foo"], [" foo"]),
            ("prefix1,prefix2", ["prefix2 foo"], [" foo"]),
            ("prefix1,prefix2", ["foo"], ["foo"]),
            ("prefix1,prefix2", ["prefix1 foo prefix1"], [" foo prefix1"]),
        ],
    )
    def test_remove_prefixes(
        self, prefixes: str, mock_completion_list: List, expected_completion_list: List
    ):
        """Test remove_prefixes."""
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            remove_prefixes=prefixes,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_remove_prefixes(text=input))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "prompt_prefix, mock_completion_list, expected_completion_list",
        [
            ("<prompt_prefix>", ["<prompt_prefix>foo"], ["foo"]),
            (
                "<prompt_prefix, has commas>",
                ["<prompt_prefix, has commas>foo"],
                ["foo"],
            ),
            ("<prompt_prefix>", ["<other_prefix>foo"], ["<other_prefix>foo"]),
            ("", ["foo"], ["foo"]),
        ],
    )
    def test_remove_prompt_prefix(
        self,
        prompt_prefix: str,
        mock_completion_list: List,
        expected_completion_list: List,
    ) -> None:
        """Test remove_prompt_prefix."""
        data = {"prompt": prompt_prefix}
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            remove_prompt_prefix=True,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_remove_prompt_prefix(text=input, data=data))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "prompt_prefix, mock_completion_list, expected_completion_list",
        [
            ("<prompt_prefix>", ["<prompt_prefix>foo"], ["<prompt_prefix>foo"]),
        ],
    )
    def test_remove_prompt_noop_if_remove_prompt_prefix_false(
        self,
        prompt_prefix: str,
        mock_completion_list: List,
        expected_completion_list: List,
    ) -> None:
        """Test when remove_prompt_prefix is set to False."""
        data = {"prompt": prompt_prefix}
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            remove_prompt_prefix=False,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_remove_prompt_prefix(text=input, data=data))
        assert output == expected_completion_list

    @pytest.mark.parametrize(
        "prompt_prefix, mock_completion_list, expected_completion_list",
        [
            ("<prompt_prefix>", ["<prompt_prefix>foo"], ["<prompt_prefix>foo"]),
        ],
    )
    def test_remove_prompt_prefix_noop_if_wrong_key_in_data(
        self,
        prompt_prefix: str,
        mock_completion_list: List,
        expected_completion_list: List,
    ):
        """Check if wrong key is provided when remove_prompt_prefix is set to True."""
        data = {"wrong_prompt_key": prompt_prefix}
        sys.path.append(get_src_dir())
        from aml_benchmark.inference_postprocessor import inference_postprocessor as inferpp

        obj = inferpp.InferencePostprocessor(
            prediction_dataset=Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE,
            prediction_column_name="prediction",
            remove_prompt_prefix=True,
        )
        output = []
        for input in mock_completion_list:
            output.append(obj.apply_remove_prompt_prefix(text=input, data=data))
        assert output == expected_completion_list

    def test_if_prediction_dataset_empty(self):
        """Test the exceptions raise if prediction dataset is empty."""
        empty_dataset_error_mssg = (
            "No .jsonl file found."
        )
        src_dir = get_src_dir()
        prediction_column_name = 'prediction'
        template = '{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}'
        dummy_dataset = os.path.join(os.getcwd(), "dummy_prediction_dataset.jsonl")
        os.system(f"touch {dummy_dataset}")
        try:
            argss = " ".join(
                [
                    "--prediction_dataset",
                    dummy_dataset,
                    "--prediction_column_name",
                    prediction_column_name,
                    "--template",
                    f"'{template}'",
                ]
            )
            cmd = f"cd {src_dir} && python -m aml_benchmark.inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, empty_dataset_error_mssg)
