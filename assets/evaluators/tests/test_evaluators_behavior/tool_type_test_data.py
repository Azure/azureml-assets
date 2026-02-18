# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test data for tool-type-specific evaluator tests.

Provides conversation data (query, response, tool_definitions, tool_calls) for each tool type:
- Function tool (user-defined, e.g., get_horoscope)
- Built-in tools: code_interpreter, bing_grounding, bing_custom_search, file_search,
  azure_ai_search, sharepoint_grounding, fabric_dataagent

Each tool type data is organized as a class with static attributes for easy access.
"""

from typing import Any, Dict, List


# =============================================================================
# FUNCTION TOOL DATA (user-defined)
# =============================================================================


class FunctionToolData:
    """Test data for user-defined function tool conversations."""

    # -- Single function call: get_horoscope --

    QUERY_HOROSCOPE: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is my horoscope? I am an Aquarius."}],
        }
    ]

    TOOL_DEFINITIONS_HOROSCOPE: List[Dict[str, Any]] = [
        {
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "required": ["sign"],
            },
        }
    ]

    TOOL_CALLS_HOROSCOPE: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "name": "get_horoscope",
            "arguments": {"sign": "Aquarius"},
        }
    ]

    RESPONSE_HOROSCOPE: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check your horoscope."},
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "get_horoscope",
                    "arguments": {"sign": "Aquarius"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": "Aquarius: Next Tuesday you will befriend a baby otter.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Your horoscope says: Next Tuesday you will befriend a baby otter.",
                }
            ],
        },
    ]

    # -- Multiple function calls: get_horoscope + send_notification --

    TOOL_DEFINITIONS_MULTI: List[Dict[str, Any]] = [
        {
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "required": ["sign"],
            },
        },
        {
            "name": "send_notification",
            "description": "Send a notification message to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The notification message to send",
                    },
                },
                "required": ["message"],
            },
        },
    ]

    TOOL_CALLS_MULTI: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "name": "get_horoscope",
            "arguments": {"sign": "Aquarius"},
        },
        {
            "type": "tool_call",
            "tool_call_id": "call_2",
            "name": "send_notification",
            "arguments": {"message": "Your horoscope is ready!"},
        },
    ]

    RESPONSE_MULTI: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "get_horoscope",
                    "arguments": {"sign": "Aquarius"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": "Aquarius: Next Tuesday you will befriend a baby otter.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "send_notification",
                    "arguments": {"message": "Your horoscope is ready!"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"status": "sent"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I've checked your horoscope and sent you a notification.",
                }
            ],
        },
    ]


# =============================================================================
# BUILT-IN TOOL DATA
# =============================================================================


class BuiltInToolData:
    """Test data for Azure built-in tool conversations.

    Built-in tools have their definitions auto-injected by the evaluator via
    _BUILT_IN_DESCRIPTIONS and _BUILT_IN_PARAMS. The tool_definitions field
    should be empty (or not provided) — the evaluator adds them automatically
    when it sees a tool call with a recognized built-in tool name.
    """

    # -- code_interpreter --

    QUERY_CODE_INTERPRETER: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Can you analyze this data and create a chart?"}],
        }
    ]

    TOOL_CALLS_CODE_INTERPRETER: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_ci_1",
            "name": "code_interpreter",
            "arguments": {"input": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])"},
        }
    ]

    RESPONSE_CODE_INTERPRETER: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_ci_1",
                    "name": "code_interpreter",
                    "arguments": {"input": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_ci_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"image": "chart.png"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I've created a chart for your data."},
            ],
        },
    ]

    EXPECTED_DEFINITION_CODE_INTERPRETER: Dict[str, Any] = {
        "type": "code_interpreter",
        "description": (
            "Use code interpreter to read and interpret information from datasets, "
            "generate code, and create graphs and charts using your data. Supports "
            "up to 20 files."
        ),
        "name": "code_interpreter",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Generated code to be executed."},
            },
        },
    }

    # -- bing_grounding --

    QUERY_BING_GROUNDING: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Search the web for the latest news on AI."}],
        }
    ]

    TOOL_CALLS_BING_GROUNDING: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_bg_1",
            "name": "bing_grounding",
            "arguments": {"requesturl": "https://api.bing.microsoft.com/v7.0/search?q=latest+AI+news"},
        }
    ]

    RESPONSE_BING_GROUNDING: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_bg_1",
                    "name": "bing_grounding",
                    "arguments": {"requesturl": "https://api.bing.microsoft.com/v7.0/search?q=latest+AI+news"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bg_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"results": [{"title": "AI breakthrough", "url": "https://example.com"}]},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here are the latest AI news results."},
            ],
        },
    ]

    EXPECTED_DEFINITION_BING_GROUNDING: Dict[str, Any] = {
        "type": "bing_grounding",
        "description": "Enhance model output with web data.",
        "name": "bing_grounding",
        "parameters": {
            "type": "object",
            "properties": {
                "requesturl": {"type": "string", "description": "URL used in Bing Search API."},
            },
        },
    }

    # -- bing_custom_search --

    QUERY_BING_CUSTOM_SEARCH: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Search our company docs for the onboarding guide."}],
        }
    ]

    TOOL_CALLS_BING_CUSTOM_SEARCH: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_bcs_1",
            "name": "bing_custom_search",
            "arguments": {"requesturl": "https://api.bing.microsoft.com/v7.0/custom/search?q=onboarding+guide"},
        }
    ]

    RESPONSE_BING_CUSTOM_SEARCH: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_bcs_1",
                    "name": "bing_custom_search",
                    "arguments": {
                        "requesturl": "https://api.bing.microsoft.com/v7.0/custom/search?q=onboarding+guide"
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bcs_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"results": [{"title": "Onboarding Guide", "url": "https://docs.example.com"}]},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I found the onboarding guide for you."},
            ],
        },
    ]

    EXPECTED_DEFINITION_BING_CUSTOM_SEARCH: Dict[str, Any] = {
        "type": "bing_custom_search",
        "description": (
            "Enables agents to retrieve content from a curated subset of websites, "
            "enhancing relevance and reducing noise from public web searches."
        ),
        "name": "bing_custom_search",
        "parameters": {
            "type": "object",
            "properties": {
                "requesturl": {
                    "type": "string",
                    "description": "Search queries, along with pre-configured site restrictions or domain filters.",
                },
            },
        },
    }

    # -- file_search --

    QUERY_FILE_SEARCH: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Find the quarterly report in my uploaded files."}],
        }
    ]

    TOOL_CALLS_FILE_SEARCH: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_fs_1",
            "name": "file_search",
            "arguments": {
                "ranking_options": {"ranker": "default_2024_08_21", "score_threshold": 0.5},
            },
        }
    ]

    RESPONSE_FILE_SEARCH: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_fs_1",
                    "name": "file_search",
                    "arguments": {
                        "ranking_options": {"ranker": "default_2024_08_21", "score_threshold": 0.5},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_fs_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": [{"file_name": "Q3_Report.pdf", "score": 0.92}],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I found the quarterly report: Q3_Report.pdf."},
            ],
        },
    ]

    EXPECTED_DEFINITION_FILE_SEARCH: Dict[str, Any] = {
        "type": "file_search",
        "description": (
            "Search for data across uploaded files. "
            "A single call can return multiple results/files in the 'results' field."
        ),
        "name": "file_search",
        "parameters": {
            "type": "object",
            "properties": {
                "ranking_options": {
                    "type": "object",
                    "properties": {
                        "ranker": {"type": "string", "description": "Ranking algorithm to use."},
                        "score_threshold": {"type": "number", "description": "Threshold for search results."},
                    },
                    "description": "Ranking options for search results.",
                },
            },
        },
    }

    # -- azure_ai_search --

    QUERY_AZURE_AI_SEARCH: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Search the knowledge base for return policy information."}],
        }
    ]

    TOOL_CALLS_AZURE_AI_SEARCH: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_ais_1",
            "name": "azure_ai_search",
            "arguments": {"input": "return policy"},
        }
    ]

    RESPONSE_AZURE_AI_SEARCH: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_ais_1",
                    "name": "azure_ai_search",
                    "arguments": {"input": "return policy"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_ais_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"documents": [{"content": "Returns accepted within 30 days."}]},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Our return policy allows returns within 30 days."},
            ],
        },
    ]

    EXPECTED_DEFINITION_AZURE_AI_SEARCH: Dict[str, Any] = {
        "type": "azure_ai_search",
        "description": "Search an Azure AI Search index for relevant data.",
        "name": "azure_ai_search",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Search terms to use."},
            },
        },
    }

    # -- sharepoint_grounding --

    QUERY_SHAREPOINT_GROUNDING: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Find the HR policy document in SharePoint."}],
        }
    ]

    TOOL_CALLS_SHAREPOINT_GROUNDING: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_sp_1",
            "name": "sharepoint_grounding",
            "arguments": {"input": "HR policy document"},
        }
    ]

    RESPONSE_SHAREPOINT_GROUNDING: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_sp_1",
                    "name": "sharepoint_grounding",
                    "arguments": {"input": "HR policy document"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_sp_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"documents": [{"title": "HR Policy 2024", "url": "https://sharepoint.example.com"}]},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I found the HR Policy 2024 document in SharePoint."},
            ],
        },
    ]

    EXPECTED_DEFINITION_SHAREPOINT_GROUNDING: Dict[str, Any] = {
        "type": "sharepoint_grounding",
        "description": (
            "Allows agents to access and retrieve relevant content from Microsoft SharePoint "
            "document libraries, grounding responses in organizational knowledge."
        ),
        "name": "sharepoint_grounding",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "A natural language query to search SharePoint content.",
                },
            },
        },
    }

    # -- fabric_dataagent --

    QUERY_FABRIC_DATAAGENT: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Query the sales data from our Fabric data agent."}],
        }
    ]

    TOOL_CALLS_FABRIC_DATAAGENT: List[Dict[str, Any]] = [
        {
            "type": "tool_call",
            "tool_call_id": "call_fd_1",
            "name": "fabric_dataagent",
            "arguments": {"input": "total sales Q3 2024"},
        }
    ]

    RESPONSE_FABRIC_DATAAGENT: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_fd_1",
                    "name": "fabric_dataagent",
                    "arguments": {"input": "total sales Q3 2024"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_fd_1",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"total_sales": 1500000, "currency": "USD"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The total sales for Q3 2024 were $1,500,000."},
            ],
        },
    ]

    EXPECTED_DEFINITION_FABRIC_DATAAGENT: Dict[str, Any] = {
        "type": "fabric_dataagent",
        "description": "Connect to Microsoft Fabric data agents to retrieve data across different data sources.",
        "name": "fabric_dataagent",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Search terms to use."},
            },
        },
    }
