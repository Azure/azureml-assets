# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for action."""

import pytest
from action_analyzer.contracts.actions.action import ActionType, Action
from action_analyzer.contracts.action_sample import ActionSample
from action_analyzer.contracts.actions.low_retrieval_score_index_action import LowRetrievalScoreIndexAction
from action_analyzer.contracts.actions.metrics_violation_index_action import MetricsViolationIndexAction


@pytest.fixture
def generate_test_action():
    """Return a test action."""
    positive_samples = []
    negative_samples = []
    for i in range(30):
        positive_samples.append(ActionSample(f"positive_query_{1}",
                                             f"positive_answer_{i}",
                                             f"positive_debugging_info_{i}",
                                             f"positive_prompt_flow_input_{i}"))
        negative_samples.append(ActionSample(f"negative_query_{1}",
                                             f"negative_answer_{i}",
                                             f"negative_debugging_info_{i}",
                                             f"negative_prompt_flow_input_{i}"))
    return Action(ActionType.METRICS_VIOLATION_INDEX_ACTION,
                  "description",
                  0.95,
                  "query intention",
                  "deployment id",
                  "run id",
                  positive_samples,
                  negative_samples)


@pytest.fixture
def generate_metrics_violation_index_action():
    """Return a MetricsViolationIndexAction action."""
    positive_samples = []
    negative_samples = []
    for i in range(30):
        positive_samples.append(ActionSample(f"positive_query_{1}",
                                             f"positive_answer_{i}",
                                             f"positive_debugging_info_{i}",
                                             f"positive_prompt_flow_input_{i}"))
        negative_samples.append(ActionSample(f"negative_query_{1}",
                                             f"negative_answer_{i}",
                                             f"negative_debugging_info_{i}",
                                             f"negative_prompt_flow_input_{i}"))
    return MetricsViolationIndexAction("index id",
                                       "index content",
                                       "coherence",
                                       0.95,
                                       "query intention",
                                       "deployment id",
                                       "run id",
                                       positive_samples,
                                       negative_samples)


@pytest.fixture
def generate_low_retrieval_score_index_action():
    """Return a LowRetrievalScoreIndexAction action."""
    positive_samples = []
    negative_samples = []
    for i in range(30):
        positive_samples.append(ActionSample(f"positive_query_{1}",
                                             f"positive_answer_{i}",
                                             f"positive_debugging_info_{i}",
                                             f"positive_prompt_flow_input_{i}"))
        negative_samples.append(ActionSample(f"negative_query_{1}",
                                             f"negative_answer_{i}",
                                             f"negative_debugging_info_{i}",
                                             f"negative_prompt_flow_input_{i}"))
    return LowRetrievalScoreIndexAction("index id",
                                        "index content",
                                        "coherence",
                                        0.95,
                                        "query intention",
                                        "deployment id",
                                        "run id",
                                        positive_samples,
                                        negative_samples)


@pytest.mark.unit
class TestActions:
    """Test class for action."""

    def test_to_json(self, generate_test_action):
        """Test base class to_json()."""
        action_json = generate_test_action.to_json()

        assert action_json["Type"] == "Metrics violation index action"
        assert action_json["Description"] == "description"
        assert action_json["ConfidenceScore"] == 0.95
        assert action_json["QueryIntention"] == "query intention"
        assert action_json["DeploymentId"] == "deployment id"
        assert action_json["RunId"] == "run id"
        assert len(action_json["PositiveSamples"]) == 30
        assert len(action_json["NegativeSamples"]) == 30

    def test_to_summary(self, generate_test_action):
        """Test base class to_summary_json()."""
        summary_json = generate_test_action.to_summary_json("output")

        assert summary_json["ActionId"] == generate_test_action.action_id
        assert summary_json["Type"] == "Metrics violation index action"
        assert summary_json["Description"] == "description"
        assert summary_json["ConfidenceScore"] == 0.95
        assert summary_json["QueryIntention"] == "query intention"
        assert summary_json["CreationTime"] == generate_test_action.creation_time
        assert summary_json["FilePath"].endswith(f"actions/{generate_test_action.action_id}.json")

    @pytest.mark.parametrize(
        "max_sample_size, expected_sample_size", [
            (10, 10),
            (20, 20),
            (30, 30),
            (40, 30)
        ])
    def test_reduce_positive_sample_size(self, generate_test_action, max_sample_size, expected_sample_size):
        """Test reduce_positive_sample_size()."""
        generate_test_action.reduce_positive_sample_size(max_sample_size)
        assert len(generate_test_action.positive_samples) == expected_sample_size

    def test_to_json_low_retrieval_score_index_action(self, generate_low_retrieval_score_index_action):
        """Test LowRetrievalScoreIndexAction class to_json()."""
        action_json = generate_low_retrieval_score_index_action.to_json()

        assert action_json["Type"] == "Low retrieval score index action"
        assert action_json["Description"].startswith("The application's response quality is low due to suboptimal index retrieval.")  # noqa
        assert action_json["ConfidenceScore"] == 0.95
        assert action_json["QueryIntention"] == "query intention"
        assert action_json["DeploymentId"] == "deployment id"
        assert action_json["RunId"] == "run id"
        assert len(action_json["PositiveSamples"]) == 30
        assert len(action_json["NegativeSamples"]) == 30
        assert action_json["IndexAssetId"] == "index id"
        assert action_json["IndexContent"] == "index content"
        assert action_json["ViolatedMetrics"] == "coherence"

    def test_to_json_metrics_violation_index_action(self, generate_metrics_violation_index_action):
        """Test MetricsViolationIndexAction class to_json()."""
        action_json = generate_metrics_violation_index_action.to_json()

        assert action_json["Type"] == "Metrics violation index action"
        assert action_json["Description"].startswith("The application's response quality is low due to suboptimal index retrieval.")  # noqa
        assert action_json["ConfidenceScore"] == 0.95
        assert action_json["QueryIntention"] == "query intention"
        assert action_json["DeploymentId"] == "deployment id"
        assert action_json["RunId"] == "run id"
        assert len(action_json["PositiveSamples"]) == 30
        assert len(action_json["NegativeSamples"]) == 30
        assert action_json["IndexAssetId"] == "index id"
        assert action_json["IndexContent"] == "index content"
        assert action_json["ViolatedMetrics"] == "coherence"
