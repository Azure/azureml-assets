import os

import pandas as pd
import pytest

from src.batch_score.common.configuration.client_settings import ClientSettingsKey
from src.batch_score.common.parallel.adjustment import AIMD, RequestMetrics
from src.batch_score.common.parallel.congestion import (
    CongestionDetector,
    CongestionState,
    RequestMetrics,
)

adjustment_tests= [
    # Test inputs format:
    # [
    #     initial_concurrency,
    #     congestion_states_to_return,
    #     expected_new_concurrency_values,
    # ]

    # If congestion state is unknown or saturated, concurrency remains the same.
    [
        1,
        10 * [CongestionState.UNKNOWN],
        10 * [1],
    ],
    [
        100,
        5 * [CongestionState.SATURATED],
        5 * [100],
    ],

    # If no congestion detected, increase concurrency.
    # - First increase is 2.
    # - Second increase is 3.
    # - Third increase is 4.
    # - All subsequent increases are 4.
    [
        1,
        7 * [CongestionState.FREE],
        [3, 6, 10, 14, 18, 22, 26],
    ],

    # Same behavior when initial concurrency is high.
    [
        1001,
        7 * [CongestionState.FREE],
        [1003, 1006, 1010, 1014, 1018, 1022, 1026],
    ],

    # TODO: Add tests when initial concurrency is zero or negative.

    # If congested, reduce concurrency.
    # But concurrency doesn't fall below 1.
    [
        100,
        15 * [CongestionState.CONGESTED],
        [75, 56, 42, 31, 23, 17, 12, 9, 6, 4, 3, 2, 1, 1, 1],
    ],
    [ 1, 5 * [CongestionState.CONGESTED], 5 * [1]],

    # Component starts. Concurrency overshoots, then stabilizes.
    [
        1,

        5 * [CongestionState.UNKNOWN]
        + 5 * [CongestionState.FREE]
        + 2 * [CongestionState.CONGESTED]
        + 5 * [CongestionState.SATURATED],

        5 * [1]
        + [3, 6, 10, 14, 18]
        + [13, 9]
        + 5 * [9],
    ],

    # Component is running optimally and then congestion is detected.
    # Concurrency drops to relieve congestion.
    [
        18,

        5 * [CongestionState.SATURATED]
        + 2 * [CongestionState.CONGESTED]
        + 5 * [CongestionState.SATURATED],

        5 * [18]
        + [13, 9]
        + 5 * [9],
    ],

    # Component is running optimally and congestion falls below saturation.
    # Concurrency increases to reach saturation.
    [
        18,

        5 * [CongestionState.SATURATED]
        + 2 * [CongestionState.FREE]
        + 5 * [CongestionState.SATURATED],

        5 * [18]
        + [20, 23]
        + 5 * [23],
    ],
]
adjustment_tests = [['make_routing_client', *test] for test in adjustment_tests]
@pytest.mark.parametrize(
    "make_routing_client, initial_concurrency, congestion_states_to_return, expected_new_concurrency_values",
    adjustment_tests,
    indirect=['make_routing_client'],
)
def test_calculate_next_concurrency(make_routing_client, initial_concurrency, congestion_states_to_return, expected_new_concurrency_values):
    aimd = AIMD(request_metrics=RequestMetrics(), client_settings_provider=make_routing_client())

    # TODO: introduce DI to accept congestion detector in constructor
    aimd._AIMD__congestion_detector = MockCongestionDetector(congestion_states_to_return)

    actual_new_concurrency_values = []
    concurrency = initial_concurrency
    for _ in congestion_states_to_return:
        adjustment = aimd.calculate_next_concurrency(concurrency)
        concurrency = adjustment.new_concurrency
        actual_new_concurrency_values.append(concurrency)

    assert expected_new_concurrency_values == actual_new_concurrency_values


fixture_names = [
    'make_routing_client',
    'mock_get_client_setting',
]

env_vars = {
    "BATCH_SCORE_CONCURRENCY_ADJUSTMENT_INTERVAL": "1000",
    "BATCH_SCORE_CONCURRENCY_INCREASE_AMOUNT": "50",
    "BATCH_SCORE_CONCURRENCY_DECREASE_RATE": "0.25",
}

@pytest.mark.parametrize(
    "make_routing_client, mock_get_client_setting, client_settings, expected_adjustment_interval, expected_additive_increase, expected_multiplicative_decrease",
    [
        (
            *fixture_names,
            {},
            180,
            1.0,
            0.75,
        ),
        (
            *fixture_names,
            {
                ClientSettingsKey.CONCURRENCY_ADJUSTMENT_INTERVAL: "60",
            },
            60,
            1.0,
            0.75,
        ),
        (
            *fixture_names,
            {
                ClientSettingsKey.CONCURRENCY_ADJUSTMENT_INTERVAL: "60",
                ClientSettingsKey.CONCURRENCY_ADDITIVE_INCREASE: "2",
                ClientSettingsKey.CONCURRENCY_MULTIPLICATIVE_DECREASE: "0.5",
            },
            60,
            2.0,
            0.5,
        ),
    ],
    indirect=fixture_names,
)
def test_init_env_var_overrides_client_setting_overrides_class_default(make_routing_client, mock_get_client_setting, client_settings, expected_adjustment_interval, expected_additive_increase, expected_multiplicative_decrease):
    # Class defaults are used when there are no client settings or environment variables.
    clear_env_vars()

    aimd_class_default = AIMD(request_metrics=RequestMetrics(), client_settings_provider=make_routing_client())

    assert 180 == aimd_class_default._AIMD__adjustment_interval
    assert 1 == aimd_class_default._AIMD__additive_increase
    assert 0.75 == aimd_class_default._AIMD__multiplicative_decrease

    # Client settings override class defaults.
    for key, value in client_settings.items():
        mock_get_client_setting[key] = value

    aimd_client_setting = AIMD(request_metrics=RequestMetrics(), client_settings_provider=make_routing_client())

    assert expected_adjustment_interval == aimd_client_setting._AIMD__adjustment_interval
    assert expected_additive_increase == aimd_client_setting._AIMD__additive_increase
    assert expected_multiplicative_decrease == aimd_client_setting._AIMD__multiplicative_decrease

    # Environment variables override client settings.
    os.environ.update(env_vars)

    aimd_env_var = AIMD(request_metrics=RequestMetrics(), client_settings_provider=make_routing_client())

    assert 1000 == aimd_env_var._AIMD__adjustment_interval
    assert 50 == aimd_env_var._AIMD__additive_increase
    assert 0.25 == aimd_env_var._AIMD__multiplicative_decrease

def clear_env_vars():
    for key in env_vars.keys():
        if key in os.environ:
            del os.environ[key]

class MockCongestionDetector(CongestionDetector):
    def __init__(self, congestion_states_to_return: 'list[CongestionState]') -> None:
        self.congestion_states_to_return = congestion_states_to_return.copy()

    def detect(self, request_metrics: RequestMetrics, start_time: pd.Timestamp, end_time: pd.Timestamp = None) -> CongestionState:
        return self.congestion_states_to_return.pop(0)
