# Known Issues – Agent Evaluator E2E Tests

This document tracks consistent errors, platform limitations, and flaky
evaluator results observed during E2E evaluation runs (both Python and .NET).

> **Last validated:** March 2026, across 3 full runs each in Python and .NET.

---

## 1. Computer Use – All Evaluators Error

**Tool type:** `computer_call`
**Affected evaluators:** All 12 evaluators
**Status:** Expected error (documented in `expected_errors` parameter)

### Details

The `computer_call` arguments format (e.g. `{ "type": "click", "x": 225, "y": 51 }`)
is not recognized by the evaluator preprocessor, which fails with:
`"The 'arguments' field must be a dictionary in tool_call content items."`
(error code `FAILED_EXECUTION`).

The preprocessor error fires **before** the unsupported-tool check, so even
the 5 evaluators with `check_for_unsupported_tools = True` get this error
instead of `NOT_APPLICABLE`. All 12 evaluators are placed in `expected_errors`.

| Evaluator              | Error Code          | Error Message                                                                |
|------------------------|---------------------|------------------------------------------------------------------------------|
| coherence, fluency, groundedness, relevance, intent_resolution, task_adherence, task_completion, tool_call_success, tool_input_accuracy, tool_output_utilization | FAILED_EXECUTION | The 'arguments' field must be a dictionary in tool_call content items. |
| tool_call_accuracy, tool_selection | FAILED_EXECUTION | Tool definitions input is required but not provided. |

---

## 2. OpenAPI – tool_call_accuracy and tool_selection Error

**Tool type:** `openapi_call`
**Affected evaluators:** tool_call_accuracy, tool_selection
**Status:** Expected error (documented in `expected_errors` parameter)

Both evaluators have `is_tool_definition_required = True`. OpenAPI tool
definitions are not provided in the evaluation input format, so they fail
with `"Tool definitions input is required but not provided."` before
reaching the unsupported-tool check.

---

## 3. Tests Requiring Missing Environment Variables

| E2E Test File                          | Missing Env Vars                                                                 |
|----------------------------------------|----------------------------------------------------------------------------------|
| test_agent_to_agent_evaluation.py      | `A2A_PROJECT_CONNECTION_ID`                                                      |
| test_azure_function_evaluation.py      | `STORAGE_INPUT_QUEUE_NAME`, `STORAGE_OUTPUT_QUEUE_NAME`, `STORAGE_QUEUE_SERVICE_ENDPOINT` |

---

## 4. Memory Search – Intermittent Content Filter Errors

**Tool type:** `memory_search`
**Affected evaluators:** task_adherence, task_completion (intermittent)
**Status:** Flaky — test uses `@pytest.mark.flaky(reruns=3)` / `[RetryFact(MaxRetries = 3)]`

Memory recall prompts occasionally trigger Azure OpenAI's content management
policy with a "jailbreak detected" filter result (`ResponsibleAIPolicyViolation`).
Non-deterministic; does not reproduce on every run.

---

## 5. Expected Quality Failures

Evaluators in `expected_failures` consistently score below threshold.
The test semantics are **inverted**: listed evaluators MUST fail — if
they unexpectedly pass, the test fails.

| Tool Type             | Evaluator              | Python | .NET  | Reason                                                |
|-----------------------|------------------------|--------|-------|-------------------------------------------------------|
| SharePoint Grounding  | relevance              | 0/6    | 0/1   | Tool may not find matching documents                  |
| SharePoint Grounding  | task_completion        | 0/6    | 0/1   | "Not found" response penalized                        |
| Fabric Data Agent     | relevance              | 0/6    | 0/1   | Tool may not find data in workspace                   |
| Fabric Data Agent     | intent_resolution      | 0/6    | 0/1   | "Not found" response penalized                        |
| Fabric Data Agent     | task_completion        | 0/6    | 0/1   | "Not found" response penalized                        |
| OpenAPI               | task_adherence         | 0/6    | 0/1   | openapi_call not recognized as tool usage             |
| Image Generation      | intent_resolution      | 0/6    | 0/1   | Evaluator can't view generated images                 |
| Image Generation      | task_completion        | 0/6    | 0/1   | Evaluator can't view generated images                 |
| Image Generation      | tool_input_accuracy    | 2/6    | 0/1   | Default params not in query penalized                 |
| Web Search            | task_adherence         | 0/6    | 0/1   | web_search_call not visible in TOOL_CALLS             |

### Cross-Platform Inconsistencies

| Tool Type | Evaluator              | Python  | .NET   | Notes                                                  |
|-----------|------------------------|---------|--------|--------------------------------------------------------|
| MCP       | tool_output_utilization| 0/6 fail| 1/1 pass| Fails consistently in Python (score=0.0), passes in .NET (score=1.0). Currently in expected_failures for both — causes .NET test failure ("expected to fail but PASSED"). TODO: remove from .NET expected_failures or investigate root cause. |
| KB MCP    | groundedness           | 5/6 pass| 0/1 fail| Flaky in Python but mostly passes; failed in .NET run (score=2). |
| KB MCP    | tool_call_success      | 5/6 pass| 0/1 fail| Flaky in Python but mostly passes; failed in .NET run (score=0). |
| MemorySearch | task_completion     | 6/6 pass| 0/1 fail| Consistently passes in Python; failed in .NET run (score=0). |

### Borderline / Flaky (not in expected_failures, handled by retry)

| Tool Type    | Evaluator         | Python Pass Rate | .NET Pass Rate | Notes                                  |
|--------------|-------------------|------------------|----------------|----------------------------------------|
| KB MCP       | groundedness      | 5/6              | 0/1            | Flaky — needs more .NET data           |
| KB MCP       | tool_call_success | 5/6              | 0/1            | Flaky — needs more .NET data           |
| MemorySearch  | task_completion  | 6/6              | 0/1            | May be flaky in .NET only              |

---

## 6. NOT_APPLICABLE by Design (Unsupported Tool Types)

**Evaluators:** tool_call_accuracy, tool_input_accuracy,
tool_output_utilization, tool_call_success, groundedness

**Tool types:** code_interpreter_call, bing_grounding, bing_custom_search,
azure_ai_search, sharepoint_grounding, azure_fabric, openapi_call,
web_search, browser_automation, computer_call

Error message: `"{tool_name} tool call is currently not supported for {evaluator_name} evaluator."`
