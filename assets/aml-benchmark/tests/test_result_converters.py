# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the functionality of the prompt factory which powers the prompt crafter."""

import json
import hashlib
import os
import pytest
import re
import sys
import tempfile
import pandas as pd

from pandas._testing.asserters import assert_frame_equal
from .test_utils import get_src_dir


sys.path.append(get_src_dir())
print(get_src_dir())

from aml_benchmark.batch_inference_preparer.endpoint_data_preparer import EndpointDataPreparer  # noqa: E402
from aml_benchmark.batch_output_formatter import main  # noqa: E402
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
            model_type, "metadata_key", "data_id_key", label_col, None, None, "fallback_value")
        expect_label = label_col if label_col else "ground_truth"
        assert rc.ground_truth_column_name == expect_label

    @pytest.mark.parametrize(
        'model_type,additional_columns', [('oai', 'a, b2'),
                                          ('oss', 'a'),
                                          ('vision_oss', None),
                                          ('claude', None)]
    )
    def test_additional_columns(self, model_type, additional_columns):
        """Test label column name."""
        rc = ResultConverters(
            model_type, "metadata_key", "data_id_key", None, additional_columns, None, "fallback_value")
        if additional_columns:
            elements = additional_columns.split(",")
            expect_additional_columns = [s.strip() for s in elements if s.strip()]
        else:
            expect_additional_columns = None
        assert rc.additional_columns == expect_additional_columns

    @pytest.mark.parametrize('has_ground_truth', [True, False])
    def test_e2e_claude(self, has_ground_truth):
        """Test parsing claude output."""
        with tempfile.TemporaryDirectory() as d:
            with tempfile.TemporaryDirectory() as gt_d:
                input_file = os.path.join(d, 'input.jsonl')
                json_str = (
                    '{"status": "success", "request": '
                    '{"prompt": "\\n\\nHuman:story of two dogs\\n\\nAssistant:", "max_tokens_to_sample": 100}, '
                    '"response": {"completion": " Here is a short story about two dogs:\\n\\nRex and Charlie '
                    'were best friends. They did everything together. They played fetch in the park, took '
                    'naps in the sun, and shared meals side by side. At night, they slept curled up next '
                    'to each other, keeping each other warm and safe. \\n\\nOne day, Rex noticed Charlie '
                    'seemed sad. His tail drooped and he moved slower than usual. \\"What\'s wrong?\\" '
                    'Rex asked. Charlie whined, \\"I", "stop_reason": "max_tokens", "stop": null}, "start": '
                    '1701718765743.0286, "end": 1701718769229.678, "latency": 3486.64927482605}')
                with open(input_file, 'w') as fp:
                    fp.write(json_str)
                prediction_data = os.path.join(d, 'results.jsonl')
                predict_ground_truth_data = os.path.join(d, 'results_ground_truth.jsonl')
                ground_truth_input = None
                if has_ground_truth:
                    ground_truth_input = os.path.join(gt_d, 'ground_truth.jsonl')
                    data = json.loads(json_str)
                    payload = hashlib.sha256(
                        re.sub(
                            r'[^A-Za-z0-9 ]+', '', data['request']['prompt']).encode('utf-8')).hexdigest()
                    pd.DataFrame({
                        EndpointDataPreparer.PAYLOAD_HASH: [payload],
                        EndpointDataPreparer.PAYLOAD_GROUNDTRUTH: ['Forty two']
                    }).to_json(ground_truth_input, orient='records', lines=True)
                main.main(
                    batch_inference_output=d,
                    model_type="claude",
                    data_id_key=None,
                    metadata_key=None,
                    label_key=None,
                    ground_truth_input=ground_truth_input,
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
                if has_ground_truth:
                    truth = 'Forty two'
                else:
                    truth = ''
                assert_frame_equal(
                    pd.read_json(
                        predict_ground_truth_data, orient='records', lines=True),
                    pd.DataFrame({ResultConverters.DEFAULT_GROUND_TRUTH: [truth]}))
