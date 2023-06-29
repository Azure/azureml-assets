# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for autoprompt constants."""


class TaskTypes:
    """Class for sharing TaskTypes."""

    arithmetic = "arithmetic"
    mcq = "multiple_choice"
    abstractive = "abstractive"


ALL_TASKS = [TaskTypes.arithmetic, TaskTypes.mcq, TaskTypes.abstractive]

GENERAL_COT_PROMPTS = [
    "Let's think step by step.",
    "First,",
    "Let's think about this logically.",
    "Let's solve this problem by splitting it into steps.",
    "Let's be realistic and think step by step.",
    "Let's think like a detective step by step.",
    "Let's think",
    "Before we dive into the answer",
    "The answer is after the proof"
]

GENERAL_SQUAD_PROMPTS = [
    '"You are an AI assistant for helping users answering question given a specific context.' +
    "You are given a context and you'll be asked a question based on the context." +
    'Your answer should be as precise as possible and answer should be only from the context.' +
    "Your answer should be succinct.\"",
    '"You are an AI assistant for helping users answering question given a specific context.' +
    "You are given a context and you'll be asked a question based on the context." +
    'Your answer should be as precise as possible and answer should be only from the context."' +
    "If you don't know the answer, say 'I don't know.'\"",
    '"You are an AI assistant for helping users answering question given a specific context.' +
    "You are given a context and you'll be asked a question based on the context." +
    'Your answer should be as precise as possible and answer should be only from the context."' +
    "Be creative in your answer.\""
]

ALL_METRICS = [
    "f1_score",
    "bert_f1",
    "bert_precision",
    "bert_recall",
    "exact_match",
    "gpt_similarity",
    "gpt_consistency",
    "gpt_relevance",
    "gpt_fluency",
    "gpt_coherence"
]

META_PROMPTS_PREFIX = "Paraphrase the sentence in a different way." + \
    "All the generated variations should be different, there should not be any duplicate answers:\n"

DEFAULT_FALLBACK_METRICS = "bert_f1"

APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"


class Activities:
    """Activity Names."""

    MAIN = "AutoPrompt"
    DATA_LOADER = "LoadingData"
    ABSTRACTIVE = "AbstractiveQAAutoPrompt"
    ARITHMETIC = "ArithmeticQAAutoPrompt"
    MCQ = "MultipleChoiceQAAutoPrompt"
    GENERATE_PROMPT = "GeneratePrompts"
    EVALUATE_PROMPTS = "EvaluatePrompts"
    GENERATE_PREDICTIONS = "GenerateTestPredictions"
    LOG_RESULTS = "LogResults"


class ExceptionLiterals:
    """Exception Literals."""

    AutoPromptTarget = "AutoPrompt"
    MetricsPackageTarget = "ComputeMetrics"
    ArgumentValidationTarget = "ArgumentValidation"
    DataLoaderTarget = "DataLoader"


class ErrorTypes:
    """Error Types."""

    Unclassified = "Unclassified"
    UserError = "UserError"
    SystemError = "SystemError"


class ErrorStrings:
    """Error Strings."""

    GenericAutoPromptError = "AutoPrompt Failed with exception [{error}]"
    GenericComputeMetricsError = "Metrics computation Failed with exception [{error}]"
    InvalidTaskType = "Task type [{task_type}] not supported."
    GenericOpenAIError = "Encountered an error in openai module. Check above traceback."
    OpenAIInitError = "Failed to fetch openai api keys with error [{error}]"
    InvalidQuestionsKey = "Questions key [{questions_key}] not found in dataset."
    InvalidAnswersKey = "Answers key [{answers_key}] not found in dataset."
    InvalidContextKey = "Context Key [{context_key}] not found in dataset."
    BadTestData = "Failed to load input dataset with error [{error}]"
