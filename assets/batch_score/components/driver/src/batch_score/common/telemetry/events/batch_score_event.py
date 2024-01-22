# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from abc import ABC
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from azureml.core import Run
from . import event_utils
from ... import constants
from ...common_enums import AuthenticationType, ApiType, EndpointType

# TODO: Add comments to describe each field
@dataclass
class BatchScoreEvent(ABC):

    @property
    def name(self):
        return "BatchScore"

    # Allow event time to be set via constructor. Helps with unit testing.
    event_time: datetime = field(init=True, default=None)
    execution_mode: str = field(init=False, default=None)
    component_name: str = field(init=False, default=None)
    component_version: str = field(init=False, default=None)
    subscription_id: str = field(init=False, default=None)
    resource_group: str = field(init=False, default=None)
    workspace_id: str = field(init=False, default=None)
    workspace_location: str = field(init=False, default=None)
    workspace_name: str = field(init=False, default=None)
    run_id: str = field(init=False, default=None)
    parent_run_id: str = field(init=False, default=None)
    experiment_id: str = field(init=False, default=None)
    node_id: str = field(init=False, default=None)
    process_id: str = field(init=False, default=None)
    endpoint_type: EndpointType = field(init=False, default=None)
    authentication_type: AuthenticationType = field(init=False, default=None)
    api_type: ApiType = field(init=False, default=None)
    async_mode: bool = field(init=False, default=None)

    def __post_init__(self) -> None:
        run = Run.get_context()
        self.run_id = run._run_id
        self.parent_run_id=run.parent.id if hasattr(run, "parent") and hasattr(run.parent, "id") else None
        self.experiment_id=run.experiment.id
        self.subscription_id = run.experiment.workspace.subscription_id
        self.resource_group = run.experiment.workspace.resource_group
        self.workspace_id = run.experiment.workspace._workspace_id_internal
        self.workspace_location = run.experiment.workspace.location
        self.workspace_name = run.experiment.workspace._workspace_name

        if self.event_time is None:
            self.event_time = datetime.now(timezone.utc)

        self.api_type = event_utils.get_api_type()
        self.async_mode = event_utils.get_async_mode()
        self.authentication_type = event_utils.get_authentication_type()
        self.component_name = event_utils.get_component_name()
        self.component_version = event_utils.get_component_version()
        self.endpoint_type = event_utils.get_endpoint_type()

        # TODO: How to get values for 'execution_mode', 'process_id' and 'node_id'?

    def __str__(self) -> str:
        return ', '.join(f'{key}: {str(value)}' for key, value in self.to_dictionary().items())

    def to_dictionary(self, include_empty_keys=True) -> dict:
        if include_empty_keys:
            return asdict(self)
        return asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})