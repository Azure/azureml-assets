import os

import pandas as pd
import pytest

from src.batch_score.common.configuration.client_settings import ClientSettingsKey
from src.batch_score.common.parallel.congestion import (
    CongestionState,
    RequestMetrics,
    WaitTimeCongestionDetector,
)
from src.batch_score.common.scoring.scoring_result import ScoringResultStatus

fixture_names = [
    'make_routing_client',
    'mock_get_client_setting',
]

env_vars = {
    "BATCH_SCORE_CONGESTION_THRESHOLD_P90_WAITTIME": "100",
    "BATCH_SCORE_SATURATION_THRESHOLD_P90_WAITTIME": "50",
}

class TestWaitTimeCongestionDetector:
    @pytest.mark.parametrize(
        "make_routing_client, mock_get_client_setting, client_settings, expected_congestion_threshold, expected_saturation_threshold",
        [
            (
                *fixture_names,
                {},
                10.0,
                5.0,
            ),
            (
                *fixture_names,
                {
                    ClientSettingsKey.CONGESTION_THRESHOLD_P90_WAIT_TIME: "99",
                },
                99.0,
                5.0,
            ),
            (
                *fixture_names,
                {
                    ClientSettingsKey.CONGESTION_THRESHOLD_P90_WAIT_TIME: "99",
                    ClientSettingsKey.SATURATION_THRESHOLD_P90_WAIT_TIME: "49",
                },
                99.0,
                49.0,
            ),
        ],
        indirect=fixture_names,
    )
    def test_init_env_var_overrides_client_setting_overrides_class_default(self, make_routing_client, mock_get_client_setting, client_settings, expected_congestion_threshold, expected_saturation_threshold):
        # Class defaults are used when there are no client settings or environment variables.
        self._clear_env_vars()

        detector_class_default = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        assert 10.0 == detector_class_default._WaitTimeCongestionDetector__congestion_threshold
        assert 5.0 == detector_class_default._WaitTimeCongestionDetector__saturation_threshold

        # Client settings override class defaults.
        for key, value in client_settings.items():
            mock_get_client_setting[key] = value

        detector_client_setting = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        assert expected_congestion_threshold == detector_client_setting._WaitTimeCongestionDetector__congestion_threshold
        assert expected_saturation_threshold == detector_client_setting._WaitTimeCongestionDetector__saturation_threshold

        # Environment variables override client settings.
        os.environ.update(env_vars)

        detector_env_var = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        assert 100 == detector_env_var._WaitTimeCongestionDetector__congestion_threshold
        assert 50 == detector_env_var._WaitTimeCongestionDetector__saturation_threshold

        # Clear env vars so subsequent tests are not affected.
        self._clear_env_vars()

    def test_unknown_congestion_case(self, make_routing_client):
        congestion_detector = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        actual_congestion_state = congestion_detector.detect(RequestMetrics(), pd.Timestamp.utcnow())

        assert actual_congestion_state == CongestionState.UNKNOWN
    
    @pytest.mark.parametrize(
        "make_routing_client, response_code, expected_congestion_state",
        # For context, see https://dev.azure.com/msdata/Vienna/_workitems/edit/2832428
        # and the TODO in WaitTimeCongestionDetector.detect().
        [('make_routing_client', r, CongestionState.UNKNOWN) for r in [400, 401, 403, 404, 500, 502, 503, 504]]
        + [('make_routing_client', ScoringResultStatus.FAILURE, CongestionState.UNKNOWN)]
        + [('make_routing_client', ScoringResultStatus.SUCCESS, CongestionState.FREE)],
        indirect=['make_routing_client'],
    )
    def test_detect_ignores_non_success_responses(self, make_routing_client, response_code, expected_congestion_state):
        congestion_detector = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        start_time = pd.Timestamp.utcnow()

        test_request_metrics = RequestMetrics()

        for i in range(5):
            self._add_result(test_request_metrics, i, 0, 1, response_code=response_code)

        actual_congestion_state = congestion_detector.detect(test_request_metrics, start_time)

        assert actual_congestion_state == expected_congestion_state


    def test_free_congestion_case(self, make_routing_client):
        congestion_detector = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        start_time = pd.Timestamp.utcnow()

        test_request_metrics = RequestMetrics()

        for i in range(5):
            self._add_result(test_request_metrics, i, 0, 1)

        actual_congestion_state = congestion_detector.detect(test_request_metrics, start_time)

        assert actual_congestion_state == CongestionState.FREE

    def test_saturated_congestion_case(self, make_routing_client):
        congestion_detector = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        start_time = pd.Timestamp.utcnow()

        test_request_metrics = RequestMetrics()

        for i in range(5):
            self._add_result(test_request_metrics, i, 1, 5)

        actual_congestion_state = congestion_detector.detect(test_request_metrics, start_time)

        assert actual_congestion_state == CongestionState.SATURATED

    def test_congested_congestion_case(self, make_routing_client):
        congestion_detector = WaitTimeCongestionDetector(client_settings_provider=make_routing_client())

        start_time = pd.Timestamp.utcnow()

        test_request_metrics = RequestMetrics()

        for i in range(5):
            self._add_result(test_request_metrics, i, 10, 20)

        actual_congestion_state = congestion_detector.detect(test_request_metrics, start_time)

        assert actual_congestion_state == CongestionState.CONGESTED

    def _add_result(self, request_metrics, request_id, additional_wait_time=10, request_total_wait_time=20, response_code=ScoringResultStatus.SUCCESS):
        request_metrics.add_result(
            request_id=request_id,
            response_code=response_code,
            response_payload="response_payload",
            model_response_code="200",
            model_response_reason="model_response_reason",
            additional_wait_time=additional_wait_time,
            request_total_wait_time=request_total_wait_time,
        )

    def _clear_env_vars(self):
        for key in env_vars.keys():
            if key in os.environ:
                del os.environ[key]