# Known Issues – Agent Evaluator E2E Tests

This document tracks consistent errors, platform limitations, and flaky
evaluator results observed during E2E evaluation runs (both Python and .NET).

> **Last validated:** July 2026 (Python).

> **2026-07-15 service capability update:** `tool_call_accuracy` and
> `tool_input_accuracy` are now supported for **all** tool types (previously
> NOT_APPLICABLE), and `tool_call_success` / `tool_output_utilization` are now
> supported for `azure_ai_search`, `azure_fabric` and `image_generation`. As a
> result, `UNSUPPORTED_TOOL_EVALUATORS` was narrowed to
> `{groundedness, tool_call_success, tool_output_utilization}` and the per-tool
> `expected_failures` were converted to `tolerated_failures` because the
> parameter-level / quality scores of the now-supported evaluators vary run to
> run. Sections below marked *(historical)* describe the pre-update behavior.

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

## 2. OpenAPI – tool_call_accuracy, tool_selection, tool_input_accuracy NOT_APPLICABLE

**Tool type:** `openapi_call`
**Affected evaluators:** tool_call_accuracy, tool_selection, tool_input_accuracy
**Status:** Tolerated (`tolerated_failures`) – they return a `not_applicable`
label (score `None`), not an error.

No tool definitions are supplied for `openapi_call` in the evaluation input, so
these three evaluators return NOT_APPLICABLE. The remaining evaluators
(including `groundedness`) pass; `task_adherence` is also tolerated because
whether `openapi_call` is counted as tool usage varies run to run.

> **Test-asset fix (2026-07-15):** `assets/weather_openapi.json` previously
> declared `components.schemes` and a top-level `auth`, which are not valid
> under OpenAPI 3.x and caused the OpenAPI tool to reject agent creation with a
> `400 tool_user_error` ("Unevaluated properties are not allowed"). Both keys
> were removed.

**(historical)** Previously these evaluators errored with `"Tool definitions
input is required but not provided."`.

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

## 5. Quality Failures (now tolerated, not asserted)

**(updated 2026-07-15)** The per-tool quality / parameter evaluators below are
non-deterministic in E2E runs (the agent's tool-call parameters and "not found"
responses vary run to run), so they were moved from `expected_failures` to
`tolerated_failures`: the test passes whether they pass or fail. Combined with
`@pytest.mark.flaky(reruns=3)` this keeps the suite stable. The table below
records which evaluators have been *observed* failing per tool type.

**(historical)** The pre-update `expected_failures` semantics were **inverted**
(listed evaluators MUST fail — if they unexpectedly passed, the test failed);
this is no longer the case.

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

**Still NOT_APPLICABLE (validated 2026-07-15):**

* `groundedness` – for every tool type listed below.
* `tool_call_success` and `tool_output_utilization` – for `code_interpreter_call`,
  `bing_grounding`, `bing_custom_search`, `web_search` and `browser_automation`.

These are represented by `UNSUPPORTED_TOOL_EVALUATORS =
{groundedness, tool_call_success, tool_output_utilization}` in `conftest.py`.
`azure_ai_search`, `azure_fabric` and `sharepoint_grounding` only keep
`groundedness` NOT_APPLICABLE (the tool-call evaluators now run), and
`image_generation` has no NOT_APPLICABLE evaluators at all.

Error message: `"{tool_name} tool call is currently not supported for {evaluator_name} evaluator."`

**(historical)** Previously `tool_call_accuracy` and `tool_input_accuracy` were
also NOT_APPLICABLE for these tool types; they are now supported everywhere.
