# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""This file contains unit tests for the prompt builder class."""

import pytest
from generation_safety_quality.annotation_compute_histogram.run import (
    _PromptBuilder, GROUNDING_ANNOTATION_TEMPLATE, ANNOTATION_REQUIREMENTS)


@pytest.mark.unit
class TestPromptBuilder:
    """Test prompt builder."""

    @pytest.mark.parametrize(
        "max_tokens,max_inputs,expected_prompt_lengths",
        [(350, 10, [298, 329, 339, 395, 355, 330, 351, 354, 335, 316, 323, 353, 340, 355]),  # noqa: E501
         (500, 10, [440, 339, 395, 498, 351, 354, 464, 489, 340, 355]),
         (1000, 10, [968, 938, 810]),
         (2000, 10, [1719, 810]),
         (2000, 20, [1855, 674]),
         (3000, 20, [2342])])
    def test_prompt_builder(
            self,
            test_data,
            max_tokens,
            max_inputs,
            expected_prompt_lengths):
        """Test prompt builder."""
        prompt_builder = _PromptBuilder(
            template=GROUNDING_ANNOTATION_TEMPLATE,
            template_requirements=ANNOTATION_REQUIREMENTS["groundedness"],
            max_tokens=max_tokens,
            min_input_examples=1,
        )
        prompts = list(prompt_builder.generate_prompts(
            test_data,
            max_inputs=max_inputs,
        ))
        assert [p.n_tokens_estimate for p in prompts] == \
            expected_prompt_lengths
        assert sum([len(p.input_examples) for p in prompts]) == len(test_data)
