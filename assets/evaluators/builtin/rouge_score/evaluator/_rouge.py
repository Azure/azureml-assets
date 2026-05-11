# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Re-export from azure-ai-evaluation SDK (functionally equivalent as of 1.16.7)
from azure.ai.evaluation import RougeScoreEvaluator
from azure.ai.evaluation._evaluators._rouge import RougeType

__all__ = ["RougeScoreEvaluator", "RougeType"]
