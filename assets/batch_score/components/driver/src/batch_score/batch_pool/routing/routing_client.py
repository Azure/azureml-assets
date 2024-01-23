import asyncio
import os
import random
from datetime import datetime, timedelta, timezone

import aiohttp

from ...batch_pool.routing.routing_utils import RoutingResponseType, classify_response
from ...common import constants
from ...common.configuration.client_settings import (
    ClientSettingsKey,
    ClientSettingsProvider,
)
from ...common.scoring.scoring_request import ScoringRequest
from ...common.telemetry import logging_utils as lu
from ...common.telemetry.logging_utils import get_events_client
from ...header_handlers.meds.meds_header_handler import MedsHeaderHandler
from ...utils.common import get_base_url


class InvalidPoolRoutes(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RoutingClient(ClientSettingsProvider):
    def __init__(
            self,
            service_namespace: str,
            target_batch_pool: str,
            header_handler: MedsHeaderHandler,
            request_path: str):
        self.__BASE_URL = os.environ.get(
            "BATCH_SCORE_ROUTING_BASE_URL",
            "https://model-endpoint-discovery.api.azureml.ms/modelendpointdiscovery")
        self.__REFRESH_INTERVAL: int = 5 * 60  # in seconds
        self.__RETRY_INTERVAL: int = 10  # in seconds
        self.__MAX_RETRY: int = 5  # maximum number of tries to get latest pool routes

        self.__service_namespace: str = service_namespace
        self.target_batch_pool: str = target_batch_pool
        self.__request_path = request_path
        self.__header_handler = header_handler

        self.__quota_scope: str = None
        self.__last_refresh: float = None
        self.__pool_routes_future: asyncio.Future = None
        self.__client_settings: dict[str, str] = {}

        self.__LIST_ROUTES_URL = (f"{self.__BASE_URL}/v1.0"
                                  f"/serviceNamespaces/{self.__service_namespace}"
                                  f"/endpointPools/{self.target_batch_pool}"
                                  f"/listEndpoints")

        # Endpoint mapping to its target percentage
        self.__target_distribution_percentages: dict[str, float] = {}

        # Endpoint mapping to in-flight request count
        self.__current_distribution_counts: dict[str, int] = {}

        # Endpoint mapping to in-flight request cost
        self.__current_distribution_costs: dict[str, int] = {}

    async def refresh_pool_routes(self, session: aiohttp.ClientSession):
        retry_count = 0

        # Retry a certain number of times or while we have no __target_distribution_percentages
        while retry_count < self.__MAX_RETRY or self.__target_distribution_percentages == {}:
            response_status = None
            try:
                lu.get_logger().debug("RoutingClient: Getting updated Pool Routes from '{}'."
                                      .format(self.__LIST_ROUTES_URL))
                request = {"trafficGroup": constants.TRAFFIC_GROUP}
                async with session.post(url=self.__LIST_ROUTES_URL,
                                        headers=self.__header_handler.get_headers(),
                                        json=request) as response:
                    response_status = response.status
                    if response_status == 200:
                        response_body = await response.json()

                        self.__quota_scope = response_body["quotaScope"]
                        self.__set_routing_configs(response_body["endpoints"])
                        self.__last_refresh = datetime.now(timezone.utc)
                        if "clientSettings" in response_body:
                            self.__client_settings = response_body["clientSettings"]
                            lu.get_logger().debug(f"RoutingClient: Client settings returned by "
                                                  f"Model Endpoint Discovery Service: {self.__client_settings}")
                        else:
                            lu.get_logger().warning("No client settings returned by Model Endpoint Discovery Service.")

                        lu.get_logger().debug("RoutingClient: Successfully updated routing pools")
                    else:
                        lu.get_logger().error("RoutingClient: Failed to update routing pools")
            except asyncio.TimeoutError:
                response_status = -408  # Manually attribute -408 as a tell to retry on asyncio exception
                lu.get_logger().error("RoutingClient: asyncio.TimeoutError")
            except aiohttp.ServerConnectionError:
                response_status = -408  # Manually attribute -408 as a tell to retry on this exception
                lu.get_logger().error("RoutingClient: aiohttp.ServerConnectionError")
            except InvalidPoolRoutes as e:
                lu.get_logger().error(f"RoutingClient: InvalidPoolRoutes exception raised: {str(e)}")

                if self.__last_refresh is None:
                    if retry_count < self.__MAX_RETRY:
                        lu.get_logger().debug(
                            "RoutingClient: No pre-existing routing pools are available - force retry")
                        response_status = -1
                    else:
                        response_status = 0  # Manually attribute no pool routes to 0 as a tell to stop retry
                else:
                    response_status = -500

            response_classification = classify_response(response_status=response_status)

            if response_classification == RoutingResponseType.RETRY:
                retry_count = retry_count + 1
                retry_wait = self.__RETRY_INTERVAL * retry_count

                lu.get_logger().debug(f"RoutingClient: Retrying in {retry_wait} seconds...")
                await asyncio.sleep(retry_wait)
            elif response_classification == RoutingResponseType.USE_EXISTING:
                lu.get_logger().debug("RoutingClient: Using pre-exisiting routing pools")
                self.__last_refresh = datetime.now(timezone.utc)
                return
            elif response_classification == RoutingResponseType.FAILURE:
                raise InvalidPoolRoutes("RoutingClient: Valid Pool Routes could not be obtained")
            elif response_classification == RoutingResponseType.SUCCESS:
                return

        lu.get_logger().info(
            "RoutingClient: Exhausted all {} retries, using existing routing pool.".format(self.__MAX_RETRY))
        self.__last_refresh = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        return self.__last_refresh is None or \
            (self.__last_refresh + timedelta(seconds=self.__REFRESH_INTERVAL)) <= datetime.now(timezone.utc)

    def increment(self, endpoint: str, request: ScoringRequest):
        if endpoint not in self.__current_distribution_counts:
            self.__current_distribution_counts[endpoint] = 1
            self.__current_distribution_costs[endpoint] = request.estimated_cost
        else:
            self.__current_distribution_counts[endpoint] += 1
            self.__current_distribution_costs[endpoint] += request.estimated_cost

        self.__emit_request_concurrency(endpoint)

    def decrement(self, endpoint: str, request: ScoringRequest):
        self.__current_distribution_counts[endpoint] -= 1
        self.__current_distribution_costs[endpoint] -= request.estimated_cost

        self.__emit_request_concurrency(endpoint)

    def get_client_setting(self, key: ClientSettingsKey) -> str:
        return self.__client_settings.get(key)

    async def get_quota_scope(self, session: aiohttp.ClientSession) -> str:
        await self.__check_and_refresh_pool_routes(session=session)

        return self.__quota_scope

    async def get_target_endpoint(self, session: aiohttp.ClientSession, exclude_endpoint: str = None) -> str:
        await self.__check_and_refresh_pool_routes(session=session)

        if len(self.__target_distribution_percentages) == 0:
            raise Exception("__target_distribution_percentages is empty.")
        else:
            # lu.get_logger().debug(
            #    f"RoutingClient: Original target distribution percentages: {self.__target_distribution_percentages}")
            pass

        use_distribution = self.__target_distribution_percentages
        effective_distribution = self.__calc_effective_dist(self.__current_distribution_counts)

        # If there are no active requests, use __target_distribution_percentages to distribute
        # otherwise:
        if not all(value == 0 for value in self.__current_distribution_counts.values()):
            distribution_delta = {endpoint: self.__target_distribution_percentages[endpoint]
                                  - (0 if endpoint not in effective_distribution
                                     else effective_distribution[endpoint])
                                  for endpoint in self.__target_distribution_percentages}

            # If the __target_distribution_percentages and effective_distribution are equivalent,
            # use __target_distribution_percentages to distribute, otherwise:
            if not all(value == 0 for value in distribution_delta.values()):
                max_key = None
                max_value = None
                for key, value in distribution_delta.items():
                    # Don't consider an excluded endpoint, unless it's the only option
                    if not (len(distribution_delta.keys()) > 1 and exclude_endpoint == key) and \
                            (max_value is None or max_value < value):
                        max_key = key
                        max_value = value

                # lu.get_logger().debug(
                #    "RoutingClient: Calculated largest distribution delta at "
                #    f"max_key: {max_key}, max_value: {max_value}")

                # Pick max_key to 100%
                use_distribution = {max_key: 1}
        else:
            lu.get_logger().debug("RoutingClient: No active requests, using target distribution to pick endpoint")

        # lu.get_logger().info(
        #    "RoutingClient: Estimated cost distribution: {}".format(self.__current_distribution_costs))
        # lu.get_logger().info("RoutingClient: Effective distribution: {}".format(effective_distribution))
        # lu.get_logger().info("RoutingClient: Using distribution: {}".format(use_distribution))

        endpoint_base_url = self.__pick_endpoint(distribution=use_distribution)
        return f"{endpoint_base_url}/{self.__request_path}"

    def __calc_effective_dist(self, distribution: "dict[str, int]") -> "dict[str, float]":
        dist_sum: float = sum(distribution.values())
        return dict(map(lambda kv: (kv[0], 0 if dist_sum == 0 else kv[1] / dist_sum), distribution.items()))

    def __emit_request_concurrency(self, endpoint):
        events_client = get_events_client()

        active_requests = self.__current_distribution_counts.get(endpoint)
        quota_reserved = self.__current_distribution_costs.get(endpoint)
        events_client.emit_request_concurrency(endpoint, active_requests, quota_reserved)

    def __pick_endpoint(self, distribution: "dict[str, float]") -> str:
        r = random.uniform(0, 1)
        running_total = 0.0

        # Order dictionary by value in ascending order
        sorted_dist = dict(sorted(distribution.items(), key=lambda item: item[1]))
        for key, value in sorted_dist.items():
            running_total += value
            if r <= running_total:
                # lu.get_logger().info("RoutingClient: Picking endpoint '{}'".format(key))
                return key

    async def __check_and_refresh_pool_routes(self, session: aiohttp.ClientSession):
        if self.is_expired():
            lu.get_logger().debug("RoutingClient: pool has expired, refreshing...")

            if self.__pool_routes_future is None or self.__pool_routes_future.done():
                loop = asyncio.get_running_loop()
                self.__pool_routes_future = loop.create_future()

                loop.create_task(self.__refresh_pool_routes(future=self.__pool_routes_future, session=session))

            await self.__pool_routes_future
            self.__pool_routes_future.result()  # Raises exception if one is encountered

    async def __refresh_pool_routes(self, future: asyncio.Future, session: aiohttp.ClientSession):
        try:
            await self.refresh_pool_routes(session=session)
            lu.get_logger().debug("RoutingClient: pool has been refreshed")
            future.set_result(True)
        except InvalidPoolRoutes as e:
            lu.get_logger().error("RoutingClient: Fatal exception encountered when refreshing "
                                  f"Routing Client pool routes: {e}")
            future.set_exception(e)
        except Exception as e:
            lu.get_logger().error(f"RoutingClient: Failed to refresh Routing Client pool routes: {e}")
            future.set_result(False)

    def __set_routing_configs(self, pool_routes: "list[any]"):
        if len(pool_routes) == 0:
            raise InvalidPoolRoutes(message="Pool Routes is empty.")

        total_traffic_weight = sum(float(route["trafficWeight"]) for route in pool_routes)

        target_distribution_percentages: dict[str, float] = {}

        for route in pool_routes:
            endpoint = str(route["endpointUri"])
            traffic_weight = float(route["trafficWeight"])

            # Endpoint discovery API will be changed to return base url only for a target endpoint.
            # Here we are converting it to base url always so it will work with both behaviors.
            endpoint_base_url = get_base_url(endpoint)

            if total_traffic_weight == 0:
                raise InvalidPoolRoutes(message="Total traffic weight is 0")
            else:
                normalized_weight = traffic_weight / total_traffic_weight

            target_distribution_percentages[endpoint_base_url] = normalized_weight

        # Update arrays
        self.__target_distribution_percentages = target_distribution_percentages
