# Evaluator Data Formats

This document describes the data formats used by the built-in evaluators — from how prompt templates are structured, to how conversation messages are reformatted into human-readable text before being injected into those prompts.

---

## 1. Prompt Template Format

Evaluator prompts are defined in `.prompty` files. Each file contains a YAML front-matter block followed by the prompt body, which is split into **role sections** using plain `system:` and `user:` headers.

### Structure

```yaml
---
name: ExampleEvaluator
description: Evaluates something useful.
model:
  api: chat
  parameters:
    temperature: 0.0
    max_tokens: 4096
    response_format:
      type: json_object

inputs:
  query:
    type: string
  response:
    type: string

---
system:
# Instruction
## Goal
You are an expert evaluator. Assess the following response.

user:
# Data
**Query:** {{query}}
**Response:** {{response}}

# Output format
Return a JSON object with a "score" and "reason".
```

### Role Headers

Prompts use **API-style role headers** — bare `system:` and `user:` labels on their own line — to delineate which part of the prompt maps to each chat-API message role. These map directly to the `role` field in the chat completions API (`"role": "system"`, `"role": "user"`).

They are **not** the ChatML token markers `<|im_start|>` / `<|im_end|>` used at the tokenizer level by some models. The `.prompty` format operates at the API abstraction layer, not the token layer.

### Template Variables

Within the `system:` and `user:` sections, Jinja2-style `{{variable}}` placeholders reference the inputs declared in the YAML front-matter. At evaluation time, these are substituted with the actual data — including the reformatted conversation text described in the sections below.

---

## 2. Reformatted Output Formats

Raw conversation message lists (as produced by the converter) are transformed into labeled, turn-based transcripts before being injected into the prompt templates above. This section describes those reformatting rules.

### Evaluator Reformatting Summary

Each evaluator includes different elements in its reformatted output. The table below shows what each evaluator includes:

| Evaluator | Conversation History | System Messages | Tool Messages (history) | Agent Response | Tool Messages (response) | Tool Definitions |
|---|---|---|---|---|---|---|
| **Coherence** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Fluency** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **IntentResolution** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Relevance** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **TaskAdherence** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **TaskCompletion** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **ToolInputAccuracy** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **ToolSelection** | ✅ | ✅ | ✅ | ❌ | — | ❌ |
| **ToolCallAccuracy** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **ToolOutputUtilization** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> **Notes:**
>
> - **ToolCallAccuracy** reformats both the conversation history and agent response including tool calls and system messages. However, it does **not** include the agent's text response — only tool call information is used.
> - **ToolSelection** parses only the tool names from the conversation; it does not reformat the full agent response.

---

### Single-Turn Evaluation

In single-turn evaluation, the conversation history (query) and agent response are reformatted **separately**.

#### Conversation History Reformatting

The conversation history is reformatted into labeled user and agent turns. The conversation must end with a user turn.

##### Basic Output

```
User turn 1:
  What is AI?

Agent turn 1:
  AI stands for Artificial Intelligence.

User turn 2:
  Tell me more.

```

##### With System Messages

When system messages are included, they are prepended:

```
SYSTEM_PROMPT:
  You are a helpful assistant.

User turn 1:
  What is AI?

Agent turn 1:
  AI stands for Artificial Intelligence.

User turn 2:
  Tell me more.

```

##### With Tool Messages

Tool calls and tool results are inlined inside the agent turn where they occur:

```
User turn 1:
  What's the weather in Seattle?

Agent turn 1:
  [TOOL_CALL] get_weather(location="Seattle", units="fahrenheit")
  [TOOL_RESULT] Temperature: 65F, Conditions: Partly cloudy
  The weather in Seattle is 65°F and partly cloudy.

User turn 2:
  Thanks for the weather info!

```

##### Multiple Tool Calls in One Turn

When the assistant invokes multiple tools, each call is followed by its result:

```
User turn 1:
  Get weather for Seattle and New York

Agent turn 1:
  I'll check the weather for both cities.
  [TOOL_CALL] get_weather(location="Seattle")
  [TOOL_RESULT] Seattle: 65F
  [TOOL_CALL] get_weather(location="New York")
  [TOOL_RESULT] New York: 72F

User turn 2:
  Thanks for checking both cities!

```

#### Agent Response Reformatting

The agent response is reformatted into a single joined string (newline-separated).

##### Without Tool Messages (default)

Only text content from assistant messages is included. Tool calls and results are **excluded**:

```
Let me check that for you.
You have one order on file.
```

##### With Tool Messages

Tool calls and results are **included** between the assistant text messages:

```
Let me check that for you.
[TOOL_CALL] get_orders(account_number="123")
[TOOL_RESULT] [{ "order_id": "A1" }]
You have one order on file.
```

#### Formatting Rules

| Element | Format |
|---|---|
| System prompt | `SYSTEM_PROMPT:\n  <text>` |
| User turn | `User turn <N>:\n  <text>` |
| Agent turn | `Agent turn <N>:\n  <text>` |
| Tool call | `[TOOL_CALL] <func_name>(<arg1>=<val1>, <arg2>=<val2>)` |
| Tool result | `[TOOL_RESULT] <result_text>` |
| Multi-line text | Each line is indented with two spaces |
| Turn separator | Empty line (`\n`) between each turn |
| String arguments | Quoted: `key="value"` |
| None arguments | `key=None` |
| Other arguments (int, bool, etc.) | Unquoted: `key=value` |

#### Edge Cases

| Scenario | Behavior |
|---|---|
| Response is `None` or `[]` | Returns `""` |
| No assistant text extracted | Returns original response unchanged |
| Parsing exception | Returns original input unchanged |
| Input is a plain string | Returns original string unchanged |

---

### Multi-Turn Evaluation

In multi-turn (conversation-level) evaluation, the **entire conversation** is reformatted as a single unified transcript rather than separate query and response fields.

Key differences from single-turn reformatting:

- **Always includes** system/developer messages, tool calls, and tool results (no opt-in flags)
- The conversation can **end with an agent turn** (no requirement for a trailing user turn)
- Consecutive messages of the same role are grouped into a single turn

#### Output Format

```
SYSTEM_PROMPT:
  You are a helpful travel assistant.

User turn 1:
  Book me a flight from NYC to London for next Friday.

Agent turn 1:
  I found several options. The cheapest is a direct British Airways flight at $450.

User turn 2:
  That works, but can you also find a hotel near ExCeL London for 3 nights?

Agent turn 2:
  [TOOL_CALL] search_hotels(location="ExCeL London", check_in="next_friday", nights=3)
  [TOOL_RESULT] Found: Holiday Inn at $120/night, 0.3 miles from ExCeL
  I found a Holiday Inn 0.3 miles from ExCeL London at $120/night for 3 nights.
```

#### Formatting Rules

| Element | Format |
|---|---|
| System/developer prompt | `SYSTEM_PROMPT:\n  <text>` |
| User turn | `User turn <N>:\n  <text>` |
| Agent turn | `Agent turn <N>:\n  <text>` |
| Tool call | `[TOOL_CALL] <func_name>(<arg1>=<val1>, <arg2>=<val2>)` |
| Tool result | `[TOOL_RESULT] <result_text>` |
| Multi-line text | Each line is indented with two spaces |
| Consecutive same-role messages | Grouped into a single turn |
| Trailing agent turn | Included (unlike single-turn reformatting) |
