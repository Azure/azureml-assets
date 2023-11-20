# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for online endpoint model."""


from typing import Optional
from ..logging import get_logger
from .endpoint_utils import EndpointUtilities


logger = get_logger(__name__)


class OnlineEndpointModel:
    """Class for online endpoint model."""
    AOAI_ENDPOINT_URL_BASE = ['openai.azure.com', 'api.cognitive.microsoft.com']

    def __init__(
            self, model: str, model_version: Optional[str], model_type: str,
            endpoint_url: Optional[str] = None
    ):
        """Init method."""
        if model is not None and model.startswith('azureml:'):
            self._model_name = model.split('/')[-3]
            self._model_path = model
            if model_version is None:
                self._model_version = model.split('/')[-1]
        else:
            self._model_name = model
            self._model_path = None
            self._model_version = model_version
        self._model_type = model_type
        if model_type is None:
            self._model_type = self._get_model_type_from_url(endpoint_url)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Get the model version."""
        return self._model_version

    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._model_type

    @property
    def model_path(self) -> str:
        """Get the model path."""
        if self._model_path is None and self.is_oss_model():
            self._model_path = 'azureml://registries/azureml-meta/models/{}/versions/{}'.format(
                self._model_name,
                self._model_version
            )
        return self._model_path

    def is_aoai_model(self) -> bool:
        """Check if the model is aoai model."""
        return self._model_type == 'oai'

    def is_oss_model(self) -> bool:
        """Check if the model is llama model."""
        return self._model_type == 'oss'

    def is_vision_oss_model(self) -> bool:
        """Check if the model is a vision oss model."""
        return self._model_type == "vision_oss"

    def _get_model_type_from_url(self, endpoint_url: str) -> str:
        if endpoint_url is None:
            logger.warning('Endpoint url is None. Default to oss.')
            return 'oss'
        for base_url in OnlineEndpointModel.AOAI_ENDPOINT_URL_BASE:
            if base_url in endpoint_url:
                return 'oai'
        return 'oss'

    @staticmethod
    def from_deployment_config_file(dir_path: str) -> "OnlineEndpointModel":
        """Get the online endpoint model from deployment config file."""
        config = EndpointUtilities.load_endpoint_metadata_json(dir_path)
        return OnlineEndpointModel(
            config['model_path'],
            None,
            config['model_type'],
            config['scoring_url']
        )
