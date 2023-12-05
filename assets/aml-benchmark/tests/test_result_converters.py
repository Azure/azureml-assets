# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the functionality of the prompt factory which powers the prompt crafter."""

import os
import pytest
import sys
import tempfile

import pandas as pd

from .test_utils import get_src_dir
from pandas._testing.asserters import assert_frame_equal


sys.path.append(get_src_dir())
print(get_src_dir())


from aml_benchmark.batch_output_formatter import main
from aml_benchmark.batch_output_formatter.result_converters import ResultConverters  # noqa: E402


class TestResultConverters:
    """Test result converters."""

    @pytest.mark.parametrize(
        'model_type,label_col', [('oai', 'label'),
                                 ('oss', None),
                                 ('vision_oss', None),
                                 ('claude', None)]
    )
    def test_label_column_name(self, model_type, label_col):
        """Test label column name."""
        rc = ResultConverters(
            model_type, "metadata_key", "data_id_key", label_col, None, "fallback_value")
        expect_label = label_col if label_col else "ground_truth"
        assert rc.ground_truth_column_name == expect_label

    def test_e2e_claude(self):
        """Test parsing claude output."""
        with tempfile.TemporaryDirectory() as d:
            input_file = os.path.join(d, 'input.jsonl')
            with open(input_file, 'w') as fp:
                fp.write(
                    '{"status": "success", "request": '
                    '{"prompt": "\\n\\nHuman:story of two dogs\\n\\nAssistant:", "max_tokens_to_sample": 100}, '
                    '"response": {"completion": " Here is a short story about two dogs:\\n\\nRex and Charlie '
                    'were best friends. They did everything together. They played fetch in the park, took '
                    'naps in the sun, and shared meals side by side. At night, they slept curled up next '
                    'to each other, keeping each other warm and safe. \\n\\nOne day, Rex noticed Charlie '
                    'seemed sad. His tail drooped and he moved slower than usual. \\"What\'s wrong?\\" '
                    'Rex asked. Charlie whined, \\"I", "stop_reason": "max_tokens", "stop": null}, "start": '
                    '1701718765743.0286, "end": 1701718769229.678, "latency": 3486.64927482605}')
            prediction_data = os.path.join(d, 'results.jsonl')
            predict_ground_truth_data = os.path.join(d, 'results_ground_truth.jsonl')
            main.main(
                batch_inference_output=d,
                model_type="claude",
                data_id_key=None,
                metadata_key=None,
                label_key=None,
                ground_truth_input=None,
                prediction_data=prediction_data,
                perf_data=os.path.join(d, 'perf.json'),
                predict_ground_truth_data=predict_ground_truth_data,
                handle_response_failure=False,
                fallback_value="wrong",
                is_performance_test=False,
                endpoint_url=('https://bedrock-runtime.us-east-1.'
                              'amazonaws.com/model/anthropic.claude-v2/invoke')
            )

            assert os.path.isfile(prediction_data)
            assert os.path.isfile(predict_ground_truth_data)
            expected_data = pd.DataFrame({
                ResultConverters.PREDICTION_COL_NAME: [
                    ' Here is a short story about two dogs:\n\nRex and Charlie '
                    'were best friends. They did everything together. They played fetch in the park, took '
                    'naps in the sun, and shared meals side by side. At night, they slept curled up next '
                    'to each other, keeping each other warm and safe. \n\nOne day, Rex noticed Charlie '
                    'seemed sad. His tail drooped and he moved slower than usual. \"What\'s wrong?\" '
                    'Rex asked. Charlie whined, \"I'
                ]
            })
            prediction_data_df = pd.read_json(prediction_data, orient='records', lines=True)
            assert_frame_equal(prediction_data_df, expected_data)
            expected_data[ResultConverters.DEFAULT_GROUND_TRUTH] = ''
            assert_frame_equal(
                pd.read_json(
                    predict_ground_truth_data, orient='records', lines=True),
                pd.DataFrame({ResultConverters.DEFAULT_GROUND_TRUTH: ['']}))
