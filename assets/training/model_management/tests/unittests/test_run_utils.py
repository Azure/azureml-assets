# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test run details."""

import platform
import sys
import unittest
from unittest.mock import MagicMock
from azureml.core import Run, Workspace
from azureml.model.mgmt.utils.logging_utils import JobRunDetails, CustomDimensions
from azureml.model.mgmt.config import AppName, LoggerConfig


class TestJobRunUtilities(unittest.TestCase):
    """Test run details."""

    def setUp(self):
        """Set up JobRunDetails object."""
        self.run = MagicMock(spec=Run)
        self.workspace = MagicMock(spec=Workspace)
        self.run.get_context.return_value = self.run
        self.run.id = "mock_run_id"
        self.run.get_details.return_value = {
            "target": "mock_compute",
            "properties": {"azureml.moduleid": "mock_asset_id"},
        }
        self.run.experiment.workspace = self.workspace
        self.workspace.name = "mock_workspace_name"
        self.run.experiment.id = "mock_experiment_id"
        self.workspace.subscription_id = "mock_subscription_id"
        self.workspace.location = "mock_region"
        parent = self.run.parent = MagicMock(spec=Run)
        parent.id = "mock_parent_run_id"
        parent.parent = None

        sys.argv.append("--model-id")
        sys.argv.append("mock_model_id")
        sys.argv.append("--model-source")
        sys.argv.append("Huggingface")
        sys.argv.append("--task-name")
        sys.argv.append("fill-mask")
        sys.argv.append("--mlflow-flavor")
        sys.argv.append("transformers")

    def test_run_details_properties(self):
        """Test run details with mock object."""
        run_details = JobRunDetails()
        run_details._run = self.run

        self.assertEqual(run_details.run_id, "mock_run_id")
        self.assertEqual(run_details.parent_run_id, "mock_parent_run_id")
        self.assertEqual(
            run_details.details, {"target": "mock_compute", "properties": {"azureml.moduleid": "mock_asset_id"}}
        )
        self.assertEqual(run_details.workspace, self.workspace)
        self.assertEqual(run_details.workspace_name, "mock_workspace_name")
        self.assertEqual(run_details.experiment_id, "mock_experiment_id")
        self.assertEqual(run_details.subscription_id, "mock_subscription_id")
        self.assertEqual(run_details.region, "mock_region")
        self.assertEqual(run_details.compute, "mock_compute")
        self.assertEqual(run_details.vm_size, None)
        self.assertEqual(run_details.component_asset_id, "mock_asset_id")
        self.assertEqual(run_details.root_attribute, "mock_parent_run_id")

        run_details._details = {"target": "mock_compute2"}
        self.assertEqual(run_details.details, {"target": "mock_compute2"})
        self.assertEqual(run_details.compute, "mock_compute2")
        self.assertEqual(run_details.component_asset_id, LoggerConfig.ASSET_NOT_FOUND)

    def test_offline_run_details_properties(self):
        """Test mock details for offline run."""
        run_details = JobRunDetails()
        run_details._run = Run.get_context()

        assert run_details.run_id.startswith("OfflineRun_")
        self.assertEqual(run_details.parent_run_id, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.details, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.workspace, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.workspace_name, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.experiment_id, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.subscription_id, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.region, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.compute, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.vm_size, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.component_asset_id, LoggerConfig.OFFLINE_RUN_MESSAGE)
        self.assertEqual(run_details.root_attribute, LoggerConfig.OFFLINE_RUN_MESSAGE)

    def test_custom_dimensions(self):
        """Test custom dimensions."""
        run_details = JobRunDetails()
        run_details._run = self.run
        custom_dimensions = CustomDimensions(run_details)
        self.assertEqual(custom_dimensions.run_id, "mock_run_id")
        self.assertEqual(custom_dimensions.parent_run_id, "mock_parent_run_id")
        self.assertEqual(custom_dimensions.workspace_name, "mock_workspace_name")
        self.assertEqual(custom_dimensions.experiment_id, "mock_experiment_id")
        self.assertEqual(custom_dimensions.subscription_id, "mock_subscription_id")
        self.assertEqual(custom_dimensions.region, "mock_region")
        self.assertEqual(custom_dimensions.compute_target, "mock_compute")
        self.assertEqual(custom_dimensions.vm_size, None)
        self.assertEqual(custom_dimensions.component_asset_id, "mock_asset_id")
        self.assertEqual(custom_dimensions.root_attribution, "mock_parent_run_id")
        self.assertEqual(custom_dimensions.os_info, platform.system())
        self.assertEqual(custom_dimensions.app_name, AppName.IMPORT_MODEL)
