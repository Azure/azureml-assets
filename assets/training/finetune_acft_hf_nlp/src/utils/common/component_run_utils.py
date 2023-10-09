# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing utility functions of the run."""

from azureml.core import Run

from azureml.acft.common_components import get_logger_app

from utils.constants import ComponentConstants


logger = get_logger_app(__name__)


class ComponentRunUtils:
    _RUN = Run.get_context()

    def get_component_asset_id(self) -> str:
        """Fetch the component name from runDto properties."""
        try:
            if isinstance(self._RUN, Run):
                run_details = self._RUN.get_details()
                return run_details['runDefinition']['componentConfiguration']['componentIdentifier']
            else:
                logger.info("Found offline run")
                return ComponentConstants.COMPONENT_ASSET_ID_NOT_FOUND
        except Exception as e:
            logger.info(f"Could not fetch the component asset id: {e}")
            return ComponentConstants.COMPONENT_ASSET_ID_NOT_FOUND

    def get_model_asset_id(self) -> str:
        """Read the model asset id from runDto properties."""
        try:
            if isinstance(self._RUN, Run):
                run_details = self._RUN.get_details()
                return run_details['runDefinition']['inputAssets']['mlflow_model_path']['asset']['assetId']
            else:
                logger.info("Found offline run")
                return ComponentConstants.MODEL_ASSET_ID_NOT_FOUND
        except Exception as e:
            logger.info(f"Could not fetch the model asset id: {e}")
            return ComponentConstants.MODEL_ASSET_ID_NOT_FOUND
