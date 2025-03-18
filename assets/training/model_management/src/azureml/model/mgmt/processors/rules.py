"""
hf_rules.py

This module defines mapping rules for Hugging Face tasks to canonical task names
and model families. Each rule is a dictionary with the following keys:
    - name: A unique identifier for the rule.
    - match: A callable that takes (task, model_id, params) and returns True if the rule applies.
    - canonical_task: The canonical task name if the rule matches.
    - model_family: The model family if the rule matches (can be a literal or a callable).
    
New rules can be added here without changing the core converter lookup logic.
"""

from azureml.model.mgmt.processors.pyfunc.config import (
    ModelFamilyPrefixes,
    SupportedTasks as PyFuncSupportedTasks,
)
from azureml.model.mgmt.processors.pyfunc.text_to_image.config import SupportedTextToImageModelFamily

HUGGINGFACE_RULES = [
    {
        "name": "mask_generation",
        "match": lambda task, model_id, params: task == PyFuncSupportedTasks.MASK_GENERATION.value,
        "canonical_task": "mask_generation",
        "model_family": None,
    },
    {
        "name": "hibou_b",
        "match": lambda task, model_id, params: (
            task == PyFuncSupportedTasks.IMAGE_FEATURE_EXTRACTION.value and 
            model_id.startswith(ModelFamilyPrefixes.HIBOU_B.value)
        ),
        "canonical_task": "hibou_b",
        "model_family": ModelFamilyPrefixes.HIBOU_B.value,
    },
    {
        "name": "virchow",
        "match": lambda task, model_id, params: task == PyFuncSupportedTasks.IMAGE_FEATURE_EXTRACTION.value,
        "canonical_task": "virchow",
        "model_family": None,
    },
    # Additional rules can be added here. For example:
    # {
    #     "name": "custom_rule",
    #     "match": lambda task, model_id, params: <condition>,
    #     "canonical_task": "<canonical_name>",
    #     "model_family": <model_family_or_callable>,
    # },
]

def get_huggingface_rule(task, model_id, params):
    """
    Iterates through HUGGINGFACE_RULES and returns the first matching rule
    as a tuple (canonical_task, model_family). If no rule matches, returns (task, None).
    
    Args:
        task (str): The raw task identifier.
        model_id (str): The model identifier.
        params (dict): Additional parameters.
        
    Returns:
        tuple: (canonical_task, model_family)
    """
    for rule in HUGGINGFACE_RULES:
        if rule["match"](task, model_id, params):
            model_family = rule["model_family"]
            if callable(model_family):
                model_family = model_family(task, model_id, params)
            return rule["canonical_task"], model_family
    return task, None
