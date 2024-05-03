# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for action sample."""

import pytest
from action_analyzer.contracts.action_sample import ActionSample, IndexActionSample


@pytest.mark.unit
class TestActionSample:
    """Test class for action sample."""

    def test_to_json_base(self):
        """Test base class to_json()."""
        action_sample = ActionSample("test question", "test answer", "test debugging info", "test prompt flow input")
        action_sampe_json = action_sample.to_json()

        assert action_sampe_json["Question"] == "test question"
        assert action_sampe_json["Answer"] == "test answer"
        assert action_sampe_json["DebuggingInfo"] == "test debugging info"
        assert action_sampe_json["PromptFlowInput"] == "test prompt flow input"

    def test_to_json_index_action_sample(self):
        """Test index action sample to_json()."""
        action_sample = IndexActionSample("test question",
                                          "test index query",
                                          "test answer",
                                          5.0,
                                          "test debugging info",
                                          "test retreival query type",
                                          3,
                                          "test prompt flow input")
        action_sampe_json = action_sample.to_json()

        assert action_sampe_json["Question"] == "test question"
        assert action_sampe_json["Answer"] == "test answer"
        assert action_sampe_json["DebuggingInfo"] == "test debugging info"
        assert action_sampe_json["PromptFlowInput"] == "test prompt flow input"
        assert action_sampe_json["IndexQuery"] == "test index query"
        assert action_sampe_json["LookupScore"] == 5.0
        assert action_sampe_json["RetrievalQueryType"] == "test retreival query type"
        assert action_sampe_json["RetrievalTopK"] == 3
