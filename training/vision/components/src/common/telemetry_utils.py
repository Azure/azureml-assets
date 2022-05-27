from azureml.automl.core.shared import log_server, logging_utilities
from azureml.automl.dnn.vision.common import utils
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.constants import SettingsLiterals

logger = get_logger('azureml.automl.dnn.vision.asset_registry')


def create_component_telemetry_wrapper(task_type):
    """Create a decorator for components that tracks telemetry."""

    def component_telemetry_wrapper(func):
        def wrapper(*args, **kwargs):
            # Initialize logging
            settings = {SettingsLiterals.TASK_TYPE: task_type}
            utils._top_initialization(settings)
            utils._set_logging_parameters(task_type, settings)

            # Set logging dimension so we can distinguish all logs coming from component.
            log_server.update_custom_dimensions({'training_component': True})

            try:
                logger.info('Training component started')
                with logging_utilities.log_activity(logger, activity_name='TrainingComponent'):
                    result = func(*args, **kwargs)
                logger.info('Training component succeeded')
                return result
            except Exception as e:
                logger.warning('Training component failed')
                logging_utilities.log_traceback(e, logger)
                raise
            finally:
                logger.info('Training component completed')
        return wrapper
    return component_telemetry_wrapper
