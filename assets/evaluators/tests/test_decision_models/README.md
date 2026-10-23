# Native Decision evaluator coverage

## OpenAI Decisions

All 19 Decision-capable assets also support the Evaluation SDK's explicit `decision_protocol="openai"`
configuration, using SDK 1.18.11 or a private build with that protocol adapter. Their published class and tag
metadata advertise `microsoft,openai`; hosted selection remains a service-approved named deployment.
Each affected asset version is incremented so a worker can require the tested protocol declaration.

The existing PiLabs/Microsoft rubrics are shared. The SDK translates them centrally to predicates and ordered
score levels; the asset's original score, applicability, threshold and composite contracts remain authoritative.
OpenAI's probability-weighted score is diagnostic, while ordinal metrics continue to use argmax.
Refusals produce explicit errors with known usage. They never become a fabricated score or skip.

`test_openai_decisions.py` packages each asset independently and checks pass/fail, paid skip, partial refusal,
actual request encoding, original answer provenance and once-only usage. Run it against the private adapter wheel
alongside the existing Decision suites. Controlled provider responses verify integration contracts, not model accuracy.

The native Decision path covers 19 model-judged built-ins. Each asset retains its
standalone inputs, score range, thresholds and declared evaluation levels. The
existing chat path remains available on supported released SDKs.

| Evaluators | Public score | Required evidence |
| --- | --- | --- |
| `fluency` | 1 to 5 | Final assistant text |
| `coherence`, `intent_resolution`, `relevance` | 1 to 5 | Query/history and response |
| `groundedness` | 1 to 5 | Response and grounding evidence; optional query |
| `task_adherence` | 0 to 1 | Query, response and applicable instructions/procedure |
| `tool_call_success` | 0 to 1 | Tool calls with matching execution results |
| `tool_input_accuracy`, `tool_output_utilization`, `tool_selection` | 0 to 1 | Query, response/tool evidence and definitions |
| `customer_satisfaction` | 1 to 5 | Query/response or conversation |
| `deflection_rate` | 0 to 1, lower is better | Response or conversation |
| `response_completeness` | 1 to 5 | Response and ground truth |
| `retrieval` | 1 to 5 | Query and ordered retrieved context, including extracted tool output |
| `similarity` | 1 to 5 | Query, response and ground truth |
| Existing: `task_completion`, `tool_call_accuracy`, `output_quality`, `tool_use_quality` | Existing standalone/composite scales | See each asset's schema |

These are native strategies within the existing assets, selected by the SDK's
Decision model configuration. Hosted callers select a named deployment. Service
profile admission, asset publication and the compatible SDK rollout are still
required before a new version is available through a deployed Foundry endpoint.
This change does not enable additional hosted evaluation levels.

Binary strategies distinguish applicability from success: applicability uses
0.5, while the configurable `probability_threshold` defaults to 0.75. An unmet
applicable requirement is a failure, not a skip. The reported public `threshold`
retains each asset's existing semantics; in particular, Task Adherence reports its
existing default threshold of 0 but passes only score 1. Deflection Rate retains
its lower-is-better score and `deflection_type` category.

Standalone prompts are authoritative. For example, standalone Intent Resolution
measures resolution, Task Adherence includes material task/procedure failures, and
Groundedness uses different definitions with and without a query. Composite
questions cannot substitute for those contracts. Groundedness conversation
evaluation now preserves separately supplied context in both chat and native
paths. Tool preprocessing also preserves caller-owned message objects.

Forced conversation evaluation preserves either mixed string/message-list
query/response orientation. Groundedness also retains a queryless response and
separately supplied context without inventing a user message. Explicit messages
continue to use their existing validation. Task Adherence's directly callable
conversation path does not change its turn-only hosted catalog declaration.
Queryless Groundedness still requires its original grounding context before any
model call. Output Quality treats an empty optional text query as absent rather
than dropping its response. These input and direct-cancellation regression
matrices are required by their owning modules in the candidate CI gate.

For column-mapped hosted requests, every explicitly referenced dataset column
must exist, even for optional evaluator inputs. Response-only Groundedness can
map an existing empty query column; direct evaluator calls may omit the query.
Neither form removes the grounding-context requirement.

Native answers provide probabilities and provenance, not generated rationales or
chat-only diagnostic fields. Paid skips retain measured token/model metadata.

## Qualification

`assets-test.yaml` builds the SDK wheel from its pinned source commit and runs the
original native suites, the three expansion suites, the packaged public
conversion matrix, and the owning Output Quality and Tool Use Quality behavior
modules. `scripts/test/check_native_sdk.py` records wheel provenance and resolves
both function and class-based JUnit cases to their owning modules. It rejects
missing modules, missing required native variants, failures or skips; cases in
another module cannot substitute for a required owning variant. The separate compatibility
job uses isolated actual releases 1.17.0, 1.18.2, 1.18.8 and 1.18.9; native requests
must reject clearly when those releases lack Decision support.

Local native command, after installing the pinned candidate wheel and test dependencies:

```shell
python -m pytest assets/evaluators/tests/test_decision_models/test_evaluators.py assets/evaluators/tests/test_decision_models/test_composites.py assets/evaluators/tests/test_decision_models/test_quality_expansion.py assets/evaluators/tests/test_decision_models/test_tool_expansion.py assets/evaluators/tests/test_decision_models/test_judge_expansion.py assets/evaluators/tests/test_decision_models/test_expansion_compatibility.py assets/evaluators/tests/test_evaluators_behavior/test_output_quality_evaluator_behavior.py assets/evaluators/tests/test_evaluators_behavior/test_tool_use_quality_evaluator_behavior.py -k "not actual_released_wheel" -q
```

The per-asset CI job runs each asset's behavior module from
`tests/test_evaluators_behavior` on the released SDK pinned in
`tests/requirements.txt`, with a 98 percent coverage gate on the asset's
`evaluator` directory. That released SDK has no Decision support, so every
expanded asset's behavior module carries released-SDK routing coverage through
`base_decision_routing_test.py`, which makes a fresh copy of the module believe
the SDK supports Decision models and asserts the routing without any transport.
The expansion suites in this directory run only in the candidate-SDK job.

Released checks require `AZUREML_COMPAT_SDK_ROOT` to contain the four isolated
`azureml-compat-sdk-<version>` installations. Without them, local compatibility
checks explicitly skip; CI provisions them and requires every case.

These controlled transport tests establish packaging, routing, parsing, score
boundaries, error handling and accounting. They do not establish real-model
judgment quality, calibration or deployed registry availability.

## Excluded from this expansion

OpenAI label/score graders and the task-level rubric skeleton need separate
contract decisions. Deterministic metrics and service-backed safety evaluators
retain their existing execution paths.
