import json

import pytest

from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.segmented_score_context import SegmentedScoreContext


@pytest.mark.parametrize("stop_reason, total_generated", [
    (None, 100),
    (None, 101),
    ("length", 100),
    ("length", 101),
    ("stop", 100),
    ("stop", 101)])
def test_has_more(
        mock_get_logger,
        mock__score_once,
        stop_reason,
        total_generated):
    max_segment_size = 2
    max_tokens = 100
    request = ScoringRequest(f'{{"prompt": "Generate something.", "max_tokens": {max_tokens}}}')
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    scoring_result = mock__score_once["scoring_result"]
    scoring_result.response_body = {"choices": [{"text": "not empty"}]}
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result)

    segmented_context._SegmentedScoreContext__last_stop_reason = stop_reason
    segmented_context._SegmentedScoreContext__total_tokens_generated = total_generated

    if stop_reason == "stop":
        assert not segmented_context.has_more()
    expected = total_generated < max_tokens
    assert segmented_context.has_more() == expected


@pytest.mark.parametrize("stop_reason", [None, "stop", "length"])
def test_has_more_no_max_tokens(
        mock_get_logger,
        mock__score_once,
        stop_reason):
    max_segment_size = 2
    request = ScoringRequest('{"prompt": "Generate something."}')
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    scoring_result = mock__score_once["scoring_result"]
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result)
    segmented_context._SegmentedScoreContext__last_stop_reason = stop_reason

    if stop_reason == "stop":
        assert not segmented_context.has_more()
    else:
        assert segmented_context.has_more()


def test_has_more_no_segmented_results(
        mock_get_logger):
    max_segment_size = 2
    request = ScoringRequest('{"prompt": "Generate something."}')
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    assert segmented_context.has_more()


@pytest.mark.parametrize("max_tokens", [1, 2])
def test_has_more_supports_segmentation_false(
        mock_get_logger,
        mock__score_once,
        max_tokens):
    max_segment_size = 2
    request = ScoringRequest(f'{{"prompt": "Generate something.", "max_tokens": {max_tokens}}}')
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    scoring_result = mock__score_once["scoring_result"]
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result)

    assert not segmented_context.has_more()


def test_has_more_when_predicted_text_is_empty(
        mock_get_logger,
        mock__score_once):
    max_segment_size = 2
    request = ScoringRequest('{"prompt": "Generate something.", "max_tokens": 3}')
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    scoring_result = mock__score_once["scoring_result"]
    scoring_result.response_body = {"choices": [{"text": ""}]}
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result)

    assert not segmented_context.has_more()


@pytest.mark.asyncio
async def test_score_next(
        make_scoring_client,
        mock_get_logger,
        mock__score_once,
        mock_get_quota_scope):
    response_body = {"id": "123",
                     "object": "text_completion",
                     "created": 456,
                     "model": "dv3",
                     "choices": [{"text": "One day", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}}

    max_segment_size = 2
    request_obj = {"prompt": "Generate something."}
    request = ScoringRequest(json.dumps(request_obj))
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    assert len(segmented_context._SegmentedScoreContext__segmented_results) == 0
    assert segmented_context._SegmentedScoreContext__next_scoring_request is None

    scoring_result = mock__score_once["scoring_result"]
    scoring_result.response_body = response_body
    scoring_result.request_obj = request_obj

    result1 = await segmented_context.score_next_once(make_scoring_client(), None)

    assert len(segmented_context._SegmentedScoreContext__segmented_results) == 1
    assert result1 == segmented_context._SegmentedScoreContext__segmented_results[0]
    assert result1.response_body == response_body
    assert segmented_context._SegmentedScoreContext__next_scoring_request is None

    request2 = segmented_context._SegmentedScoreContext__create_next_scoring_request()

    assert request2.cleaned_payload_obj['prompt'] == 'Generate something.One day'

    result2 = await segmented_context.score_next_once(make_scoring_client(), None)

    assert len(segmented_context._SegmentedScoreContext__segmented_results) == 2
    assert result2 == segmented_context._SegmentedScoreContext__segmented_results[1]
    assert segmented_context._SegmentedScoreContext__next_scoring_request is None


def test_build_scoring_result_one_segment(
        mock_get_logger,
        mock__score_once):
    max_segment_size = 2
    request_obj = {"prompt": "Generate something."}
    request = ScoringRequest(json.dumps(request_obj))
    segmented_context = SegmentedScoreContext(request, max_segment_size)

    scoring_result = mock__score_once["scoring_result"]
    scoring_result.response_body = {"choices": [{"text": "not empty"}]}
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result)

    test_value = "return input when only one segment in results"
    final_result = segmented_context.build_scoring_result(test_value)
    assert final_result == test_value


def test_build_scoring_result(
        mock_get_logger,
        mock__score_once):
    max_segment_size = 2
    request_obj = {"prompt": "Generate something."}
    request = ScoringRequest(json.dumps(request_obj))
    response_body = {"id": "123",
                     "object": "text_completion",
                     "created": 456,
                     "model": "dv3",
                     "choices": [{"text": "One day", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}}

    response_body2 = {"id": "123",
                     "object": "text_completion",
                     "created": 456,
                     "model": "dv3",
                     "choices": [{"text": "One day", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}}

    segmented_context = SegmentedScoreContext(request, max_segment_size)

    scoring_result = mock__score_once["scoring_result"]
    scoring_result.response_body = response_body
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result)

    scoring_result2 = scoring_result.copy()
    scoring_result2.response_body = response_body2
    segmented_context._SegmentedScoreContext__segmented_results.append(scoring_result2)

    final_result = segmented_context.build_scoring_result(scoring_result2)

    # Ensure the overall result is not the same as the last output from model
    assert final_result != scoring_result2

    # Ensure the last output from model did not change.
    assert segmented_context._SegmentedScoreContext__segmented_results[-1] == scoring_result2
    assert scoring_result2.response_body["usage"]["prompt_tokens"] == 5
    assert scoring_result2.response_body["usage"]["completion_tokens"] == 15
    assert scoring_result2.response_body["usage"]["total_tokens"] == 20

    # Ensure counts correct on overall result
    assert final_result.response_body["usage"]["prompt_tokens"] == 2
    assert final_result.response_body["usage"]["completion_tokens"] == 18
    assert final_result.response_body["usage"]["total_tokens"] == 20


