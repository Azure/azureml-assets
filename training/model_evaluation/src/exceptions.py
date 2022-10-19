from azureml.exceptions import AzureMLException
from constants import ExceptionLiterals 

class ModelEvaluationException(AzureMLException):
    def __init__(self, exception_message, inner_exception=None, target=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        if not target:
            target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)

class ArgumentValidationException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.ARGS_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)


class DataValidationException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.DATA_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)


class DataLoaderException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.DATA_LOADING_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)


class ModelValidationException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.MODEL_LOADER_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)


class ScoringException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)


class PredictException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)


class ComputeMetricsException(ModelEvaluationException):
    def __init__(self, exception_message, inner_exception=None, details=None, message_format=None, message_parameters=None, reference_code=None, **kwargs):
        target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message, inner_exception=inner_exception, target=target, details=details, message_format=message_format, 
                        message_parameters=message_parameters, reference_code=reference_code, **kwargs)
