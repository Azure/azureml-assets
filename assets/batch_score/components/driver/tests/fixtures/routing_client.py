import aiohttp
import pytest

from src.batch_score.batch_pool.routing.routing_client import RoutingClient


@pytest.fixture
def make_routing_client(make_completion_header_handler):
    def make(service_namespace: str = None, target_batch_pool: str = None, header_handler = make_completion_header_handler(), request_path: str = None):
        return RoutingClient(
            service_namespace=service_namespace,
            target_batch_pool=target_batch_pool,
            header_handler=header_handler,
            request_path=request_path
        )
    
    return make

@pytest.fixture
def mock_get_quota_scope(monkeypatch):
    async def _get_quota_scope(self, session):
        return "endpointPools:MOCK-POOL:trafficGroups:MOCK-GROUP"

    monkeypatch.setattr("src.batch_score.batch_pool.routing.routing_client.RoutingClient.get_quota_scope", _get_quota_scope)

@pytest.fixture
def mock_get_client_setting(monkeypatch):
    state = {}

    def _get_client_setting(self, key):
        return state.get(key)
    
    monkeypatch.setattr("src.batch_score.batch_pool.routing.routing_client.RoutingClient.get_client_setting", _get_client_setting)

    return state

@pytest.fixture
def mock_refresh_pool_routes(monkeypatch):
    state = {
        "exception_to_raise": None,
    }

    async def _refresh_pool_routes(self, session: aiohttp.ClientSession):
        if state["exception_to_raise"]:
            raise state["exception_to_raise"]

        pass
    
    monkeypatch.setattr("src.batch_score.batch_pool.routing.routing_client.RoutingClient.refresh_pool_routes", _refresh_pool_routes)

    return state
