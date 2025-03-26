from dataclasses import dataclass
from typing import Callable, Dict, List, Type, Union


@dataclass
class ConvertorRule:
    predicate: Callable[[str, dict], bool]
    factory: Union[str, Type["MLflowConvertorFactoryInterface"]]


class ConvertorFactoryRegistry:
    _rules: Dict[str, List[ConvertorRule]] = {}

    @classmethod
    def register(cls, framework: str, rule: ConvertorRule) -> None:
        cls._rules.setdefault(framework, []).append(rule)

    @classmethod
    def get_factory(
        cls, framework: str, task: str, params: dict
    ) -> Union[str, Type["MLflowConvertorFactoryInterface"]]:
        if framework not in cls._rules:
            raise ValueError(f"Unsupported model framework: {framework}")
        for rule in cls._rules[framework]:
            if rule.predicate(task, params):
                return rule.factory
        raise ValueError(
            f"No converter found for task '{task}' in framework '{framework}'."
        )


ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: SupportedNLPTasks.has_value(task), factory="NLP"
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: SupportedVisionTasks.has_value(task),
        factory="Vision",
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task
        in {
            PyFuncSupportedTasks.TEXT_TO_IMAGE.value,
            PyFuncSupportedTasks.TEXT_TO_IMAGE_INPAINTING.value,
            PyFuncSupportedTasks.IMAGE_TO_IMAGE.value,
        },
        factory=TextToImageMLflowConvertorFactory,
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task
        == SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value,
        factory=ASRMLflowConvertorFactory,
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task
        == PyFuncSupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value
        or (
            task == PyFuncSupportedTasks.EMBEDDINGS.value
            and params.get("model_id", "").startswith(ModelFamilyPrefixes.CLIP.value)
        ),
        factory="CLIP",
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task == PyFuncSupportedTasks.EMBEDDINGS.value
        and params.get("model_id", "").startswith(ModelFamilyPrefixes.DINOV2.value),
        factory="DinoV2",
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task
        in {
            PyFuncSupportedTasks.IMAGE_TO_TEXT.value,
            PyFuncSupportedTasks.VISUAL_QUESTION_ANSWERING.value,
        },
        factory="BLIP",
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task
        == PyFuncSupportedTasks.MASK_GENERATION.value,
        factory="SegmentAnything",
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.HUGGINGFACE.value,
    ConvertorRule(
        predicate=lambda task, params: task
        == PyFuncSupportedTasks.IMAGE_FEATURE_EXTRACTION.value,
        factory="Virchow",
    ),
)

# MMLAB mappings:
ConvertorFactoryRegistry.register(
    ModelFramework.MMLAB.value,
    ConvertorRule(
        predicate=lambda task, params: MMLabDetectionTasks.has_value(task),
        factory="MMLabDetection",
    ),
)
ConvertorFactoryRegistry.register(
    ModelFramework.MMLAB.value,
    ConvertorRule(
        predicate=lambda task, params: MMLabTrackingTasks.has_value(task),
        factory="MMLabTracking",
    ),
)

# LLaVA mappings:
ConvertorFactoryRegistry.register(
    ModelFramework.LLAVA.value,
    ConvertorRule(
        predicate=lambda task, params: task
        == PyFuncSupportedTasks.IMAGE_TEXT_TO_TEXT.value,
        factory="LLaVA",
    ),
)

# AutoML mappings:
ConvertorFactoryRegistry.register(
    ModelFramework.AutoML.value,
    ConvertorRule(
        predicate=lambda task, params: task
        in {
            PyFuncSupportedTasks.IMAGE_CLASSIFICATION.value,
            PyFuncSupportedTasks.IMAGE_CLASSIFICATION_MULTILABEL.value,
            PyFuncSupportedTasks.IMAGE_OBJECT_DETECTION.value,
            PyFuncSupportedTasks.IMAGE_INSTANCE_SEGMENTATION.value,
        },
        factory="AutoML",
    ),
)


def get_mlflow_convertor(
    model_framework: str,
    model_dir: str,
    output_dir: str,
    temp_dir: str,
    translate_params: dict,
):
    """
    Instantiate and return an MLflow convertor based on the model framework and task.
    """
    task = translate_params["task"]
    factory = ConvertorFactoryRegistry.get_factory(
        model_framework, task, translate_params
    )

    if isinstance(factory, str):
        # Use the base factory, which selects the converter class by a key.
        return BaseMLflowConvertorFactory(factory).create_mlflow_convertor(
            model_dir, output_dir, temp_dir, translate_params
        )
    elif isinstance(factory, type):
        # Instantiate the specialized factory class.
        return factory().create_mlflow_convertor(
            model_dir, output_dir, temp_dir, translate_params
        )
    else:
        raise ValueError(
            f"Invalid factory type for task '{task}' in framework '{model_framework}'."
        )
