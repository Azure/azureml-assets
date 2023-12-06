import pytest

from src.batch_score.common.scoring.tally_failed_request_handler import (
    TallyFailedRequestHandler,
)


@pytest.fixture()
def make_tally_failed_request_handler():
    def make(enabled: bool = False, tally_exclusions: str = None) -> TallyFailedRequestHandler:
        return TallyFailedRequestHandler(
            enabled=enabled,
            tally_exclusions=tally_exclusions
        )
    
    return make