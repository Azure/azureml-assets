# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import aiohttp
from aiohttp import TraceConfig

from ..telemetry import logging_utils as lu


class ExceptionTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_request_exception.append(self.__on_request_exception)

    async def __on_request_exception(self, session, trace_config_ctx, params: aiohttp.TraceRequestExceptionParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}ExceptionTrace exceptiom: {params.exception}")


class RequestEndTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_request_end.append(self.__on_request_end)

    async def __on_request_end(self, session, trace_config_ctx, params: aiohttp.TraceRequestEndParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}RequestEndTrace response: {params.response}")


class RequestRedirectTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_request_redirect.append(self.__on_request_redirect)

    async def __on_request_redirect(self, session, trace_config_ctx, params: aiohttp.TraceRequestRedirectParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}RequestRedirectTrace response: {params.response}")


class ResponseChunkReceivedTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_response_chunk_received.append(self.__on_response_chunk_received)

    async def __on_response_chunk_received(
            self,
            session,
            trace_config_ctx,
            params: aiohttp.TraceResponseChunkReceivedParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}ResponseChunkReceivedTrace chunk: {params.chunk}")


class ConnectionCreateStartTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_connection_create_start.append(self.__on_connection_create_start)

    async def __on_connection_create_start(
            self,
            session,
            trace_config_ctx,
            params: aiohttp.TraceConnectionCreateStartParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}ConnectionCreateStartTrace: Connection creation started")


class ConnectionCreateEndTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_connection_create_end.append(self.__on_connection_create_end)

    async def __on_connection_create_end(
            self,
            session,
            trace_config_ctx,
            params: aiohttp.TraceConnectionCreateEndParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}ConnectionCreateEndTrace: Connection creation ended")


class ConnectionReuseconnTrace(TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_connection_reuseconn.append(self.__on_connection_reuseconn)

    async def __on_connection_reuseconn(
            self,
            session,
            trace_config_ctx,
            params: aiohttp.TraceConnectionReuseconnParams):
        prefix = ""
        if trace_config_ctx.trace_request_ctx and "worker_id" in trace_config_ctx.trace_request_ctx:
            prefix = "{}: ".format(trace_config_ctx.trace_request_ctx["worker_id"])
        lu.get_logger().debug(f"{prefix}ConnectionReuseconnTrace: Connection reused")
