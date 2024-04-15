# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for online endpoint model."""


from typing import Optional
import re
from azureml.core import Run, Model
from aml_benchmark.utils.exceptions import BenchmarkUserException
from aml_benchmark.utils.error_definitions import BenchmarkUserError
from azureml._common._error_definition.azureml_error import AzureMLError

from ..logging import get_logger
from ..aml_run_utils import (
    get_dependent_run,
    get_run_name,
    get_current_step_raw_input_value,
    get_root_run
)


logger = get_logger(__name__)


class OnlineEndpointModel:
    """Class for online endpoint model."""

    AOAI_ENDPOINT_URL_BASE = [
        'openai.azure.com', 'api.cognitive.microsoft.com', 'cognitiveservices.azure.com/']

    def __init__(
            self, model: str, model_version: Optional[str], model_type: str,
            endpoint_url: Optional[str] = None,
            is_finetuned: Optional[bool] = False,
            finetuned_subscription_id: Optional[str] = None,
            finetuned_resource_group: Optional[str] = None,
            finetuned_workspace: Optional[str] = None,
            model_depend_step: Optional[str] = None,
            is_aoai_finetuned_model: Optional[bool] = False
    ):
        """Init method."""
        if model is not None and model.startswith('azureml:'):
            self._model_name = model.split('/')[-3]
            self._model_path = model
            if model_version is None:
                self._model_version = model.split('/')[-1]
        elif self._get_model_type_from_url(endpoint_url) == 'claude':
            self._model_path = endpoint_url
            self._model_name = 'anthropic.claude'
            self._model_version = re.findall(r'anthropic.claude-v(\d+[:]*\d*)/', endpoint_url)[0]
            self._model_type = 'claude'
        else:
            self._model_name = model
            self._model_path = None
            self._model_version = model_version
        self._model_type = model_type
        if model_type is None:
            self._model_type = self._get_model_type_from_url(endpoint_url)
        self._is_finetuned = is_finetuned
        self._finetuned_subscription_id = finetuned_subscription_id
        self._finetuned_resource_group = finetuned_resource_group
        self._finetuned_workspace = finetuned_workspace
        self._model_depend_step = model_depend_step
        self._is_aoai_finetuned_model = is_aoai_finetuned_model

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Get the model version."""
        if self._model_version is None and self.is_finetuned:
            logger.warning('Model version is None. Trying to get one from the run.')
            finetuned_run_id = None
            if self.model_depend_step is not None:
                finetuned_run = get_dependent_run(self.model_depend_step)
                ws = Run.get_context().experiment.workspace
                finetuned_run_id = self._get_model_registered_run_id(finetuned_run)
                logger.info(f"Finetuned run id is {finetuned_run_id}")
            models = list(Model.list(ws, self._model_name, run_id=finetuned_run_id))
            if len(models) == 0:
                raise BenchmarkUserException._with_error(
                    AzureMLError.create(
                        BenchmarkUserError,
                        error_details=f"No associate version with model name {self._model_name} in step "
                                      f"{self.model_depend_step} can be found. Please make sure the finetuned "
                                      f"step {self.model_depend_step} has successfully registered the model."
                    )
                )
            self._model_version = str(models[0].version)
            if len(models) > 1:
                logger.warning(
                    f"Multiple models with name {self._model_name} are found. "
                    f"Use first one with version  {self._model_version} now."
                )
        return self._model_version

    @property
    def model_depend_step(self) -> str:
        """Get the model depend step."""
        if self._model_depend_step is None:
            raw_input = get_current_step_raw_input_value('wait_input')
            if raw_input:
                self._model_depend_step = raw_input.split('.')[2]
        return self._model_depend_step

    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._model_type

    @property
    def is_finetuned(self) -> bool:
        """Get the finetune flag."""
        return self._is_finetuned

    @property
    def is_aoai_finetuned_model(self) -> bool:
        """Get the finetune flag."""
        return self._is_aoai_finetuned_model

    @property
    def source(self) -> Optional[str]:
        """Get the source."""
        if self.is_finetuned:
            source_list = [
                "/subscriptions", self._finetuned_subscription_id, "resourceGroups",
                self._finetuned_resource_group, "providers", "Microsoft.MachineLearningServices",
                "workspaces", self._finetuned_workspace
            ]
            return "/".join(source_list)
        return None

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

    def is_claude_model(self) -> bool:
        """Check if the model is claude model."""
        return self._model_type == 'claude'

    def is_vision_oss_model(self) -> bool:
        """Check if the model is a vision oss model."""
        return self._model_type == "vision_oss"

    def _get_model_type_from_url(self, endpoint_url: str) -> str:
        if endpoint_url is None:
            logger.warning('Endpoint url is None. Default to oss.')
            return 'oss'
        if "claude" in endpoint_url:
            return 'claude'
        for base_url in OnlineEndpointModel.AOAI_ENDPOINT_URL_BASE:
            if base_url in endpoint_url:
                return 'oai'
        return 'oss'

    @staticmethod
    def _get_model_registered_run_id(finetuned_run: Run) -> str:
        """Get the run id of the step that registered the model."""
        root_run_id = finetuned_run.properties.get('azureml.rootpipelinerunid',  get_root_run().id)
        reused_run_id = None
        try:
            step_runs = list(finetuned_run.get_children())[0].get_children()
            for s in step_runs:
                if get_run_name(s) == 'openai_completions_finetune':
                    reused_run_id = OnlineEndpointModel._get_reused_root_run_id(s)
        except Exception as e:
            logger.warning(f"The input step is not a pipeline component, using its id now. {e}")
            reused_run_id = OnlineEndpointModel._get_reused_root_run_id(finetuned_run)
        return reused_run_id if reused_run_id else root_run_id

    @staticmethod
    def _get_reused_root_run_id(run: Run) -> Optional[str]:
        if run.properties.get('azureml.isreused', "").lower() == 'true':
            return run.properties.get('azureml.rootpipelinerunid')
        return None
