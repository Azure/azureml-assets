# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test data for tool-type-specific evaluator tests.

Each tool type has QUERY, RESPONSE, and TOOL_DEFINITIONS data
converted from converter_test_output/ JSON files.
"""

# =============================================================================
# 1. Function Tool
# =============================================================================

LOCAL_CALLS_QUERY = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can use function tools.",
    },
    {
        "role": "user",
        "content": [
            {
                "text": "What is my horoscope? I am an Aquarius.",
                "type": "input_text",
            }
        ],
    },
]

LOCAL_CALLS_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "function_call",
                "tool_call_id": "call_ASUI6ResxjPRW7JDubafRBQX",
                "name": "get_horoscope",
                "arguments": {"sign": "Aquarius"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_ASUI6ResxjPRW7JDubafRBQX",
        "role": "tool",
        "content": [
            {
                "type": "function_call_output",
                "function_call_output": {
                    "horoscope": "Aquarius: Next Tuesday you will befriend a baby otter."
                },
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": "Your horoscope for Aquarius is: Next Tuesday you will befriend a baby otter.",
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

LOCAL_CALLS_TOOL_DEFINITIONS = [
    {
        "name": "get_horoscope",
        "type": "function",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius",
                }
            },
            "required": ["sign"],
            "additionalProperties": False,
        },
    }
]

# Expected tool_calls after _parse_tools_from_response extracts and enriches them
LOCAL_CALLS_EXPECTED_FLOW_TOOL_CALLS = [
    {
        "type": "tool_call",
        "tool_call_id": "call_ASUI6ResxjPRW7JDubafRBQX",
        "name": "get_horoscope",
        "arguments": {"sign": "Aquarius"},
        "tool_result": {
            "horoscope": "Aquarius: Next Tuesday you will befriend a baby otter."
        },
    }
]

# =============================================================================
# 2. Code Interpreter
# =============================================================================

CODE_INTERPRETER_QUERY = [
    {
        "role": "user",
        "content": [
            {
                "text": "write  a code that does the Fibonacci series and try it out with value 5.",
                "type": "input_text",
            }
        ],
    },
]

CODE_INTERPRETER_RESPONSE = [
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "Sure! Here\u2019s a simple Python code to generate the Fibonacci series up to the nth term "
                    "(for example, `n = 5`). I\u2019ll also run this code for `n = 5` so you see the result.\n\n"
                    "```python\ndef fibonacci_series(n):\n    fib_series = [0, 1]\n    while len(fib_series) < n:\n"
                    "        next_value = fib_series[-1] + fib_series[-2]\n        fib_series.append(next_value)\n"
                    "    return fib_series[:n]\n\n# Try it out with value 5\nresult = fibonacci_series(5)\n"
                    "print(result)\n```\n\nNow, I'll run this code for `n = 5` and show you the output:"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "ci_ed84972d9a7e33e800696a67791c448190bf7d4dbb38950626",
                "name": "code_interpreter_call",
                "arguments": {
                    "input": (
                        "def fibonacci_series(n):\n    fib_series = [0, 1]\n    while len(fib_series) < n:\n"
                        "        next_value = fib_series[-1] + fib_series[-2]\n        fib_series.append(next_value)\n"
                        "    return fib_series[:n]\n\n# Try it out with value 5\nresult = fibonacci_series(5)\nresult"
                    )
                },
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "ci_ed84972d9a7e33e800696a67791c448190bf7d4dbb38950626",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": [
                    {
                        "logs": "The 15th number in the Fibonacci sequence is 610.\n",
                        "type": "logs",
                    },
                    {
                        "logs": "'The 15th number in the Fibonacci sequence is 610.'",
                        "type": "logs",
                    },
                ],
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "The Fibonacci series for the value 5 is:  \n**[0, 1, 1, 2, 3]**\n\n"
                    "Let me know if you'd like to try it for another value or a different programming language!"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

CODE_INTERPRETER_TOOL_DEFINITIONS = [
    {
        "name": "code_interpreter_call",
        "type": "code_interpreter",
        "description": (
            "Use code interpreter to read and interpret information from datasets, "
            "generate code, and create graphs and charts using your data. Supports up to 20 files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Generated code to be executed.",
                }
            },
        },
    }
]

# =============================================================================
# 3. Bing Grounding
# =============================================================================

BING_GROUNDING_QUERY = [
    {"role": "system", "content": "Use bing"},
    {
        "role": "user",
        "content": [
            {
                "text": "what's the weather like in Cairo?",
                "type": "input_text",
            }
        ],
    },
]

BING_GROUNDING_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_a1d55ba09983404e9bb2d8e3d166483d",
                "name": "bing_grounding",
                "arguments": {"query": "current weather in Cairo"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_a1d55ba09983404e9bb2d8e3d166483d",
        "role": "tool",
        "content": [{"type": "tool_result", "tool_result": ""}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [
                    {
                        "end_index": 490,
                        "start_index": 478,
                        "title": "Cairo, Cairo, Egypt Weather Forecast | AccuWeather",
                        "type": "url_citation",
                        "url": "https://www.accuweather.com/en/eg/cairo/127164/weather-forecast/127164",
                    },
                    {
                        "end_index": 502,
                        "start_index": 490,
                        "title": "Cairo Weather Forecast",
                        "type": "url_citation",
                        "url": "https://www.weather-forecast.com/locations/Cairo/forecasts/latest",
                    },
                ],
                "text": (
                    "Right now in Cairo, the weather is warm and mostly clear. The temperature is about "
                    "76\u00b0F (24\u00b0C) in the late evening, with a high during the day around 81\u00b0F (27\u00b0C). "
                    "It\u2019s windy, with gusts up to 27 mph. The air quality is currently very unhealthy, so "
                    "outdoor activity should be limited, especially for sensitive groups. In the coming days, "
                    "expect warm and mostly sunny weather, with daytime highs ranging from the upper 70s to "
                    "low 80s\u00b0F (25\u201328\u00b0C) and cool nights around 60\u00b0F (16\u00b0C)"
                    "\u30104:0\u2020source\u3011\u30104:2\u2020source\u3011."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

BING_GROUNDING_TOOL_DEFINITIONS = [
    {
        "name": "bing_grounding",
        "type": "remote_tool",
        "parameters": {
            "knowledge_sources": [
                {
                    "connection_id": (
                        "/subscriptions/7b43cfa1-da92-48cc-865d-5499466b3b5c/resourceGroups/"
                        "Eval-Cairo-rg-WUS2/providers/Microsoft.CognitiveServices/accounts/"
                        "asayedahmed-ngen-swcentral-resou/projects/asayedahmed-ngen-swcentral/"
                        "connections/BingSearchResource2akd2v"
                    ),
                    "query_parameters": {
                        "top_k_results": 5,
                        "language": "en",
                        "market": "en-us",
                    },
                }
            ],
            "query": {"type": "string"},
        },
    }
]

# =============================================================================
# 4. Bing Custom Search
# =============================================================================

BING_CUSTOM_SEARCH_QUERY = [
    {
        "role": "system",
        "content": (
            "Use Bing Custom Search. The allowed domains (including subdomains) are: "
            "https://www.microsoft.com, https://www.bing.com, https://www.google.com, "
            "https://learn.microsoft.com"
        ),
    },
    {
        "role": "user",
        "content": [
            {
                "text": "Can you bing (as in google) the weather",
                "type": "input_text",
            }
        ],
    },
]

BING_CUSTOM_SEARCH_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_41eb4d8d783b48649ee5eb03ba4d2c98",
                "name": "bing_custom_search",
                "arguments": {"query": "weather"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_41eb4d8d783b48649ee5eb03ba4d2c98",
        "role": "tool",
        "content": [{"type": "tool_result", "tool_result": ""}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [
                    {
                        "end_index": 308,
                        "start_index": 296,
                        "title": "Us Weather Forecast 10 Day - Bing Maps",
                        "type": "url_citation",
                        "url": "https://www.bing.com/maps?q=Us+Weather+Forecast+10+Day",
                    },
                    {
                        "end_index": 320,
                        "start_index": 308,
                        "title": "7 day weather forecast - Bing Maps",
                        "type": "url_citation",
                        "url": "https://www.bing.com/maps/search?q=7+day+weather+forecast&cp=36.670567~-78.243187&lvl=11&style=r",
                    },
                    {
                        "end_index": 559,
                        "start_index": 547,
                        "title": "weather-radar - Google Maps",
                        "type": "url_citation",
                        "url": "https://www.google.com/maps/search/weather-radar",
                    },
                    {
                        "end_index": 571,
                        "start_index": 559,
                        "title": "Google Maps",
                        "type": "url_citation",
                        "url": "https://www.google.com/maps/search/local.radar+weather",
                    },
                ],
                "text": (
                    "For the latest weather forecast, the National Weather Service is a trusted source "
                    "and provides up-to-date weather information, including hourly and daily forecasts. "
                    "You can check their forecasts for your specific area on their official website: "
                    "[National Weather Service](https://www.weather.gov)"
                    "\u30104:0\u2020source\u3011\u30104:3\u2020source\u3011.\n\n"
                    "You can also use services like Bing Maps and Google Maps to view weather radar and "
                    "local forecasts directly on their platforms. These often include weather overlays, "
                    "live radar, and traffic data integrated with the map views"
                    "\u30104:2\u2020source\u3011\u30104:4\u2020source\u3011."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

BING_CUSTOM_SEARCH_TOOL_DEFINITIONS = [
    {
        "name": "bing_custom_search",
        "type": "remote_tool",
        "parameters": {
            "knowledge_sources": [
                {
                    "connection_id": (
                        "/subscriptions/7b43cfa1-da92-48cc-865d-5499466b3b5c/resourceGroups/"
                        "Eval-Cairo-rg-WUS2/providers/Microsoft.CognitiveServices/accounts/"
                        "asayedahmed-ngen-swcentral-resou/projects/asayedahmed-ngen-swcentral/"
                        "connections/bingcustomsearch2akd2v"
                    ),
                    "query_parameters": {
                        "bing_custom_search_instance_name": "general-configuration",
                        "top_k_results": 5,
                        "language": "en",
                        "market": "en-us",
                    },
                }
            ],
            "query": {"type": "string"},
        },
    }
]

# =============================================================================
# 5. File Search
# =============================================================================

FILE_SEARCH_QUERY = [
    {
        "role": "system",
        "content": "Use tool search for when user asks about the cafes/diners.",
    },
    {
        "role": "user",
        "content": [
            {"text": "what's a good restaurant", "type": "input_text"}
        ],
    },
]

FILE_SEARCH_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "fs_c742d50075e5271900698e1c5a674c8190ae2f207c496d2001",
                "name": "file_search_call",
                "arguments": {
                    "queries": [
                        "good restaurant recommendation",
                        "best restaurant",
                        "top rated restaurant",
                        "recommended restaurants",
                        "what's a good restaurant",
                    ]
                },
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "fs_c742d50075e5271900698e1c5a674c8190ae2f207c496d2001",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": [
                    {
                        "attributes": {},
                        "file_id": "assistant-StE61XCSBRyLv11Ytyckea",
                        "filename": "french_cafe_menu.md",
                        "score": 0.0333,
                        "text": "# Le Jardin de Paris",
                        "vector_store_id": "",
                    },
                    {
                        "attributes": {},
                        "file_id": "assistant-Xpu5yP1AQZiB86Hz5iG4uv",
                        "filename": "italian_diner_menu.md",
                        "score": 0.0328,
                        "text": "# Trattoria Bella Notte",
                        "vector_store_id": "",
                    },
                ],
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [
                    {
                        "file_id": "assistant-StE61XCSBRyLv11Ytyckea",
                        "filename": "french_cafe_menu.md",
                        "index": 346,
                        "type": "file_citation",
                    },
                    {
                        "file_id": "assistant-Xpu5yP1AQZiB86Hz5iG4uv",
                        "filename": "italian_diner_menu.md",
                        "index": 620,
                        "type": "file_citation",
                    },
                ],
                "text": (
                    "If you're looking for a good restaurant, here are two tasty options based on "
                    "the menus provided:\n\n1. **Le Jardin de Paris** (French Caf\u00e9)\n"
                    "2. **Trattoria Bella Notte** (Italian Diner)\n\n"
                    "Both are excellent choices\u2014pick based on whether you\u2019re in the mood "
                    "for French or Italian!"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

FILE_SEARCH_TOOL_DEFINITIONS = [
    {
        "name": "file_search_call",
        "type": "file_search",
        "description": (
            "Search for data across uploaded files. A single call can return multiple "
            "results/files in the 'results' field."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ranking_options": {
                    "type": "object",
                    "properties": {
                        "ranker": {
                            "type": "string",
                            "description": "Ranking algorithm to use.",
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Threshold for search results.",
                        },
                    },
                    "description": "Ranking options for search results.",
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
    }
]

# Expected tool_calls after _parse_tools_from_response
FILE_SEARCH_EXPECTED_FLOW_TOOL_CALLS = [
    {
        "type": "tool_call",
        "tool_call_id": "fs_c742d50075e5271900698e1c5a674c8190ae2f207c496d2001",
        "name": "file_search_call",
        "arguments": {
            "queries": [
                "good restaurant recommendation",
                "best restaurant",
                "top rated restaurant",
                "recommended restaurants",
                "what's a good restaurant",
            ]
        },
        "tool_result": [
            {
                "attributes": {},
                "file_id": "assistant-StE61XCSBRyLv11Ytyckea",
                "filename": "french_cafe_menu.md",
                "score": 0.0333,
                "text": "# Le Jardin de Paris",
                "vector_store_id": "",
            },
            {
                "attributes": {},
                "file_id": "assistant-Xpu5yP1AQZiB86Hz5iG4uv",
                "filename": "italian_diner_menu.md",
                "score": 0.0328,
                "text": "# Trattoria Bella Notte",
                "vector_store_id": "",
            },
        ],
    }
]

# =============================================================================
# 6. Azure AI Search
# =============================================================================

AZURE_AI_SEARCH_QUERY = [
    {"role": "system", "content": "Use azure search tool"},
    {
        "role": "user",
        "content": [
            {
                "text": "can you use the azure ai search tool to see the top hotels in it?",
                "type": "input_text",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "Absolutely! I can use the Azure AI Search tool to find information on the top hotels. "
                    "Please clarify what \u201cit\u201d refers to\u2014are you looking for top hotels in a "
                    "specific city, country, or another location? Let me know your preference so I can "
                    "search accurately."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "text": "in any location, just hit the tool",
                "type": "input_text",
            }
        ],
    },
]

AZURE_AI_SEARCH_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_0981aece8c424fbf9f7d1f43307f3bd8",
                "name": "azure_ai_search",
                "arguments": {"query": "top hotels"},
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [
                    {
                        "end_index": 1096,
                        "start_index": 1084,
                        "title": "doc_0",
                        "type": "url_citation",
                        "url": "https://selshafey-ai-search.search.windows.net/",
                    },
                    {
                        "end_index": 1108,
                        "start_index": 1096,
                        "title": "doc_1",
                        "type": "url_citation",
                        "url": "https://selshafey-ai-search.search.windows.net/",
                    },
                    {
                        "end_index": 1120,
                        "start_index": 1108,
                        "title": "doc_2",
                        "type": "url_citation",
                        "url": "https://selshafey-ai-search.search.windows.net/",
                    },
                    {
                        "end_index": 1132,
                        "start_index": 1120,
                        "title": "doc_3",
                        "type": "url_citation",
                        "url": "https://selshafey-ai-search.search.windows.net/",
                    },
                    {
                        "end_index": 1144,
                        "start_index": 1132,
                        "title": "doc_4",
                        "type": "url_citation",
                        "url": "https://selshafey-ai-search.search.windows.net/",
                    },
                ],
                "text": (
                    "Here are some of the top hotels identified using the Azure AI Search tool:\n\n"
                    "1. **Double Sanctuary Resort**  \n"
                    "   - 5-star luxury hotel with the largest rooms in the city, ranked #1 by "
                    "Traveler magazine. Offers free WiFi, flexible check-in/out, fitness center, "
                    "and in-room espresso.\n\n"
                    "2. **Gastronomic Landscape Hotel**\n"
                    "   - Renowned for culinary excellence, led by acclaimed chef William Dough "
                    "who oversees all dining services.\n\n"
                    "3. **Economy Universe Motel**\n"
                    "   - Local, family-run hotel in bustling downtown Redmond. Pet-friendly and "
                    "close to parks, highways, and major cities. A good budget option.\n\n"
                    "4. **Countryside Hotel**\n"
                    "   - Offers up to 50% savings compared to traditional hotels. Features free "
                    "WiFi, downtown proximity, a full kitchen, washer/dryer, and a bowling "
                    "alley\u2014great for extended stays.\n\n"
                    "5. **Bellevue Suites**\n"
                    "   - Centrally located in downtown Bellevue, this hotel offers apartment-style "
                    "suites and a free shuttle to the airport. Steps from the Light Rail for easy "
                    "city access.\n\n"
                    "Let me know if you want more details or hotels focused on a specific location "
                    "or style!"
                    "\u30106:0\u2020source\u3011\u30106:1\u2020source\u3011\u30106:2\u2020source\u3011"
                    "\u30106:3\u2020source\u3011\u30106:4\u2020source\u3011"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

AZURE_AI_SEARCH_TOOL_DEFINITIONS = [
    {
        "name": "azure_ai_search",
        "type": "remote_tool",
        "parameters": {
            "knowledge_sources": [
                {
                    "index_details": {
                        "index_configuration": {
                            "index_connection_id": (
                                "/subscriptions/7b43cfa1-da92-48cc-865d-5499466b3b5c/resourceGroups/"
                                "Eval-Cairo-rg-WUS2/providers/Microsoft.CognitiveServices/accounts/"
                                "asayedahmed-ngen-swcentral-resou/projects/asayedahmed-ngen-swcentral/"
                                "connections/selshafeyaisearch2akd2v"
                            ),
                            "index_name": "hotels-sample-index",
                        }
                    },
                    "query_parameters": {
                        "top_k_results": 5,
                        "query_type": "simple",
                    },
                }
            ],
            "query": {"type": "string"},
        },
    }
]

# =============================================================================
# 7. SharePoint Grounding
# =============================================================================

SHAREPOINT_QUERY = [
    {"role": "system", "content": "use the sharepoint tool you have"},
    {
        "role": "user",
        "content": [
            {
                "text": "hit the sharepoint tool with a search on income details",
                "type": "input_text",
            }
        ],
    },
]

SHAREPOINT_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_4e129052eab74f8d9533b307a0bba340",
                "name": "sharepoint_grounding",
                "arguments": {"query": "income details"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_4e129052eab74f8d9533b307a0bba340",
        "role": "tool",
        "content": [
            {"type": "tool_result", "tool_result": {"documents": []}}
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "I searched using the term \"income details\" but couldn't find any relevant "
                    "information in your SharePoint data."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

SHAREPOINT_TOOL_DEFINITIONS = [
    {
        "name": "sharepoint_grounding",
        "type": "remote_tool",
        "parameters": {
            "knowledge_sources": [
                {
                    "connection_id": (
                        "/subscriptions/7b43cfa1-da92-48cc-865d-5499466b3b5c/resourceGroups/"
                        "Eval-Cairo-rg-WUS2/providers/Microsoft.CognitiveServices/accounts/"
                        "asayedahmed-ngen-swcentral-resou/projects/asayedahmed-ngen-swcentral/"
                        "connections/test_sharepoint"
                    )
                }
            ],
            "query": {"type": "string"},
        },
    }
]

# =============================================================================
# 8. Fabric Data Agent
# =============================================================================

FABRIC_QUERY = [
    {"role": "system", "content": "Use Fabric tool"},
    {
        "role": "user",
        "content": [
            {
                "text": "what's the average of my income?",
                "type": "input_text",
            }
        ],
    },
]

FABRIC_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_74d8c06ed93b4beeb76cc305c35c5ad1",
                "name": "azure_fabric",
                "arguments": {"query": "average income for the user"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_74d8c06ed93b4beeb76cc305c35c5ad1",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": {
                    "documents": [
                        {
                            "id": "416ae013-64ed-4356-b633-ce9e306a74ca",
                            "content": "The user's average income is 7,071.43.\n",
                            "filepath": "fabric://response",
                            "title": "Fabric Response for: average income for the user",
                            "url": (
                                "https://msit.fabric.microsoft.com/groups/944d4cb8-9517-4219-991c-"
                                "da6c31f3c4e7/aiskills/830e03d6-4afd-484b-8328-ee9d85510307/"
                                "stage/published/threads/thread_8gXP6muTIDVIltjLPYkYGpcy/runs/"
                                "run_fabg2zOsCZiNOi15DRtcaIgwrijm/question/"
                                "msg_Fh2TVkvUgI0wwMvz9T3OzQl6/source/foundry"
                            ),
                            "metadata": (
                                '{"source":"azure_fabric","query":"average income for the user",'
                                '"data_agent_url":"https://msit.fabric.microsoft.com/groups/'
                                '944d4cb8-9517-4219-991c-da6c31f3c4e7/aiskills/'
                                '830e03d6-4afd-484b-8328-ee9d85510307/stage/published/threads/'
                                'thread_8gXP6muTIDVIltjLPYkYGpcy/runs/'
                                'run_fabg2zOsCZiNOi15DRtcaIgwrijm/question/'
                                'msg_Fh2TVkvUgI0wwMvz9T3OzQl6/source/foundry"}'
                            ),
                            "score": 1,
                            "knowledgeSourceIndex": 0,
                        }
                    ]
                },
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [
                    {
                        "end_index": 43,
                        "start_index": 31,
                        "title": "Fabric Response for: average income for the user",
                        "type": "url_citation",
                        "url": (
                            "https://msit.fabric.microsoft.com/groups/944d4cb8-9517-4219-991c-"
                            "da6c31f3c4e7/aiskills/830e03d6-4afd-484b-8328-ee9d85510307/"
                            "stage/published/threads/thread_8gXP6muTIDVIltjLPYkYGpcy/runs/"
                            "run_fabg2zOsCZiNOi15DRtcaIgwrijm/question/"
                            "msg_Fh2TVkvUgI0wwMvz9T3OzQl6/source/foundry"
                        ),
                    }
                ],
                "text": "Your average income is 7,071.43\u30104:0\u2020source\u3011.",
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

FABRIC_TOOL_DEFINITIONS = [
    {
        "name": "azure_fabric",
        "type": "remote_tool",
        "parameters": {
            "knowledge_sources": [
                {
                    "connection_id": (
                        "/subscriptions/7b43cfa1-da92-48cc-865d-5499466b3b5c/resourceGroups/"
                        "Eval-Cairo-rg-WUS2/providers/Microsoft.CognitiveServices/accounts/"
                        "asayedahmed-ngen-swcentral-resou/projects/asayedahmed-ngen-swcentral/"
                        "connections/fabric_dataagent_salmas_finance"
                    )
                }
            ],
            "query": {"type": "string"},
        },
    }
]

# =============================================================================
# 9. OpenAPI
# =============================================================================

OPENAPI_QUERY = [
    {
        "role": "user",
        "content": [
            {
                "text": "What's the weather like in Cairo?",
                "type": "input_text",
            }
        ],
    },
]

OPENAPI_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "openapi_call",
                "tool_call_id": "call_cfb55a91a58c44ea9217b34174aad2ab",
                "name": "weather_GetCurrentWeather",
                "arguments": {"location": "Cairo", "format": "j1"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_cfb55a91a58c44ea9217b34174aad2ab",
        "role": "tool",
        "content": [
            {"type": "openapi_call_output", "openapi_call_output": ""}
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "**Current weather in Cairo:**\n\n- **Temperature:** 26\u00b0C (feels like 25\u00b0C)\n"
                    "- **Condition:** Sand (likely some dusty or sandy winds)\n"
                    "- **Humidity:** 28%\n"
                    "- **Cloud Cover:** 0% (clear skies)\n"
                    "- **Wind:** SW at 23 km/h\n"
                    "- **Visibility:** Moderate (4 km)\n"
                    "- **No precipitation**\n"
                    "- **UV Index:** 2\n\n"
                    "**Summary:** Cairo is experiencing warm, dry, and sunny weather, but there is "
                    "sand or dust in the air which may reduce visibility. Skies are clear and "
                    "it\u2019s breezy. Make sure to protect yourself from the dust if you\u2019re "
                    "heading outside!"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

OPENAPI_TOOL_DEFINITIONS = [
    {
        "name": "weather_GetCurrentWeather",
        "type": "openapi",
        "description": "Get weather information for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location to retrieve the weather for",
                },
                "format": {
                    "type": "string",
                    "default": "j1",
                    "description": "Always use j1 value for this parameter",
                },
            },
            "additionalProperties": False,
            "required": ["location", "format"],
        },
    }
]

# =============================================================================
# 10. Web Search
# =============================================================================

WEB_SEARCH_QUERY = [
    {"role": "system", "content": "Use web search tool."},
    {
        "role": "user",
        "content": [
            {
                "text": "what's the weather like in Napoli, Italy?",
                "type": "input_text",
            }
        ],
    },
]

WEB_SEARCH_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "ws_11a1935fb2e04a5500697232507208819098b1ff4c651c776f",
                "name": "web_search",
                "arguments": {
                    "query": "current weather Napoli Italy",
                    "type": "search",
                    "queries": ["current weather Napoli Italy"],
                },
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [
                    {
                        "end_index": 670,
                        "start_index": 516,
                        "title": "Naples, Naples, Italy - The Weather Channel",
                        "type": "url_citation",
                        "url": (
                            "https://weather.com/en/weather/today/l/Naples+Naples+Italy?"
                            "placeId=14e0aa668cd31f8abaa8c062ee3093a077f450a190eed1b822d52bc02e5cc16d"
                        ),
                    },
                    {
                        "end_index": 737,
                        "start_index": 672,
                        "title": "Naples (Italy) weather - Met Office",
                        "type": "url_citation",
                        "url": "https://weather.metoffice.gov.uk/forecast/sr60kyq6u",
                    },
                ],
                "text": (
                    "The current weather in Napoli, Italy is partly cloudy with a temperature of "
                    "around 10\u00b0C, feeling like 7\u00b0C."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

WEB_SEARCH_TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "type": "web_search_preview",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "search_context_size": {
                    "type": "string",
                    "default": "medium",
                },
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
    }
]

# =============================================================================
# 11. Browser Automation
# =============================================================================

BROWSER_AUTOMATION_QUERY = [
    {
        "role": "system",
        "content": (
            "You are an Agent helping with browser automation tasks. "
            "You can answer questions, provide information, and assist with various tasks "
            "related to web browsing using the Browser Automation tool available to you."
        ),
    },
    {
        "role": "user",
        "content": [
            {
                "text": (
                    "Your goal is to report the percent of Microsoft year-to-date stock price change.\n"
                    "To do that, go to the website finance.yahoo.com.\n"
                    "At the top of the page, you will find a search bar.\n"
                    "Enter the value 'MSFT', to get information about the Microsoft stock price.\n"
                    "At the top of the resulting page you will see a default chart of Microsoft stock price.\n"
                    "Click on 'YTD' at the top of that chart, and report the percent value that shows up "
                    "just below it."
                ),
                "type": "input_text",
            }
        ],
    },
]

BROWSER_AUTOMATION_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_6cea93fc479841a9b15050c0e54d9f76",
                "name": "browser_automation",
                "arguments": {
                    "query": (
                        "Go to finance.yahoo.com, enter 'MSFT' in the search bar to find Microsoft "
                        "stock. Click on 'YTD' at the top of the chart and report the percent change "
                        "value displayed below it."
                    )
                },
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_6cea93fc479841a9b15050c0e54d9f76",
        "role": "tool",
        "content": [{"type": "tool_result", "tool_result": ""}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "The year-to-date (YTD) percent change for Microsoft (MSFT) stock, as shown on "
                    "Yahoo Finance, is\u202f**16.19%**."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

BROWSER_AUTOMATION_TOOL_DEFINITIONS = [
    {
        "name": "browser_automation",
        "type": "remote_tool",
        "parameters": {
            "connection_id": (
                "/subscriptions/7b43cfa1-da92-48cc-865d-5499466b3b5c/resourceGroups/"
                "Eval-Cairo-rg-WUS2/providers/Microsoft.CognitiveServices/accounts/"
                "asayedahmed-ngen-swcentral-resou/projects/asayedahmed-ngen-swcentral/"
                "connections/playright-workspace"
            ),
            "query": {"type": "string"},
        },
    }
]

# =============================================================================
# 12. Computer Use
# =============================================================================

COMPUTER_USE_QUERY = [
    {
        "role": "system",
        "content": (
            "\n        You are a computer automation assistant. \n        \n        "
            "Be direct and efficient. When you reach the search results page, read and "
            "describe the actual search result titles and descriptions you can see.\n        "
        ),
    },
    {
        "role": "user",
        "content": [
            {
                "text": (
                    "I need you to help me search for 'OpenAI news'. Please type 'OpenAI news' "
                    "and submit the search. Once you see search results, the task is complete."
                ),
                "type": "input_text",
            }
        ],
    },
]

COMPUTER_USE_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_FInoTTjI3ZVo8YABKZZNAvX0",
                "name": "computer_call",
                "arguments": None,
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_FInoTTjI3ZVo8YABKZZNAvX0",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": {
                    "type": "input_image",
                    "file_id": None,
                    "image_url": None,
                    "detail": "high",
                },
            }
        ],
    },
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_jCD5nKperYyNzZy61PJpc6HV",
                "name": "computer_call",
                "arguments": None,
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_jCD5nKperYyNzZy61PJpc6HV",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": {"type": "computer_screenshot"},
            }
        ],
    },
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_vtyWFNQC3rPYocF61Z5oF5TY",
                "name": "computer_call",
                "arguments": None,
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_vtyWFNQC3rPYocF61Z5oF5TY",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": {"type": "computer_screenshot"},
            }
        ],
    },
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_SH0b4q6oVRVMW9050de2awWS",
                "name": "computer_call",
                "arguments": None,
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "call_SH0b4q6oVRVMW9050de2awWS",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": {"type": "computer_screenshot"},
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "The search results for \"OpenAI news\" are displayed, featuring information on "
                    "recent updates and innovations in OpenAI's products and research, such as the "
                    "introduction of GPT-5 and advancements in image generation models. Do you want "
                    "any specific information from these results?"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

COMPUTER_USE_TOOL_DEFINITIONS = []

# =============================================================================
# 13. Image Generation
# =============================================================================

IMAGE_GEN_QUERY = [
    {"role": "system", "content": "Generate images based on user prompts"},
    {
        "role": "user",
        "content": [
            {
                "text": "Generate an image of Microsoft logo.",
                "type": "input_text",
            }
        ],
    },
]

IMAGE_GEN_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "ig_03423d8b2c8d106c006994475fad508193959c7c92607fee67",
                "name": "image_generation",
                "arguments": {
                    "background": "opaque",
                    "output_format": "png",
                    "quality": "low",
                    "size": "1024x1024",
                    "revised_prompt": "Microsoft logo on a white background",
                },
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "ig_03423d8b2c8d106c006994475fad508193959c7c92607fee67",
        "role": "tool",
        "content": [
            {"type": "tool_result", "tool_result": "<generated_image_data>"}
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": "Here is an image of the Microsoft logo.",
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

IMAGE_GEN_TOOL_DEFINITIONS = [
    {
        "name": "image_generation",
        "type": "image_generation",
        "description": "Generate images based on user prompts",
        "parameters": {
            "type": "object",
            "properties": {
                "background": {"type": "string", "default": "auto"},
                "input_fidelity": {"type": "string"},
                "input_image_mask": {"type": "string"},
                "moderation": {"type": "string", "default": "auto"},
                "output_compression": {"type": "integer", "default": 100},
                "output_format": {"type": "string", "default": "png"},
                "partial_images": {"type": "string"},
                "quality": {"type": "string", "default": "low"},
                "size": {"type": "string", "default": "1024x1024"},
                "n": {"type": "integer", "default": 1},
                "revised_prompt": {"type": "string"},
            },
            "additionalProperties": False,
        },
    }
]

# Expected tool_calls after _parse_tools_from_response
IMAGE_GEN_EXPECTED_FLOW_TOOL_CALLS = [
    {
        "type": "tool_call",
        "tool_call_id": "ig_03423d8b2c8d106c006994475fad508193959c7c92607fee67",
        "name": "image_generation",
        "arguments": {
            "background": "opaque",
            "output_format": "png",
            "quality": "low",
            "size": "1024x1024",
            "revised_prompt": "Microsoft logo on a white background",
        },
        "tool_result": "<generated_image_data>",
    }
]

# =============================================================================
# 14. Memory Search
# =============================================================================

MEMORY_SEARCH_QUERY = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers general questions",
    },
    {
        "role": "user",
        "content": [
            {
                "text": "Please order my usual coffee",
                "type": "input_text",
            }
        ],
    },
]

MEMORY_SEARCH_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "memupdateo_2f31edffff39a63f00699441f5791081909eaae75cd08dd354",
                "name": "memory_search",
                "arguments": {},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "memupdateo_2f31edffff39a63f00699441f5791081909eaae75cd08dd354",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": [
                    {
                        "content": "User prefers dark roast coffee.",
                        "kind": "user_profile",
                        "memory_id": "3a353f9202ca41bf95a4a9ef21d90d41",
                        "scope": "user_123",
                        "updated_at": 1771323829,
                    },
                    {
                        "content": "User prefers dark roast coffee.",
                        "kind": "user_profile",
                        "memory_id": "3a353f9202ca41bf95a4a9ef21d90d41",
                        "scope": "user_123",
                        "updated_at": 1771323829,
                    },
                    {
                        "content": (
                            "The user stated a preference for dark roast coffee. Dark roast coffee "
                            "is characterized by a bold flavor, rich aroma, and lower acidity "
                            "compared to lighter roasts."
                        ),
                        "kind": "chat_summary",
                        "memory_id": "9117c9f9d7424f0290c523d1cd3de45a",
                        "scope": "user_123",
                        "updated_at": 1771323829,
                    },
                ],
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "Sure! I'll order your usual\u2014one dark roast coffee. Would you like any "
                    "specific size or extras (milk, sugar, etc.) with that?"
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

MEMORY_SEARCH_TOOL_DEFINITIONS = [
    {
        "name": "memory_search",
        "type": "memory_search",
        "description": "Search memory for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_store_name": {
                    "type": "string",
                    "default": "my_memory_store",
                },
                "scope": {"type": "string", "default": "user_123"},
            },
            "additionalProperties": False,
        },
    }
]

# Expected tool_calls after _parse_tools_from_response
MEMORY_SEARCH_EXPECTED_FLOW_TOOL_CALLS = [
    {
        "type": "tool_call",
        "tool_call_id": "memupdateo_2f31edffff39a63f00699441f5791081909eaae75cd08dd354",
        "name": "memory_search",
        "arguments": {},
        "tool_result": [
            {
                "content": "User prefers dark roast coffee.",
                "kind": "user_profile",
                "memory_id": "3a353f9202ca41bf95a4a9ef21d90d41",
                "scope": "user_123",
                "updated_at": 1771323829,
            },
            {
                "content": "User prefers dark roast coffee.",
                "kind": "user_profile",
                "memory_id": "3a353f9202ca41bf95a4a9ef21d90d41",
                "scope": "user_123",
                "updated_at": 1771323829,
            },
            {
                "content": (
                    "The user stated a preference for dark roast coffee. Dark roast coffee "
                    "is characterized by a bold flavor, rich aroma, and lower acidity "
                    "compared to lighter roasts."
                ),
                "kind": "chat_summary",
                "memory_id": "9117c9f9d7424f0290c523d1cd3de45a",
                "scope": "user_123",
                "updated_at": 1771323829,
            },
        ],
    }
]

# =============================================================================
# 15. Knowledge Base MCP
# =============================================================================

KB_MCP_QUERY = [
    {
        "role": "user",
        "content": [{"text": "hi", "type": "input_text"}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": "Hello! How can I help you today?",
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"text": "do you have any tools?", "type": "input_text"}
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "Yes, I do! I have several tools that help me provide information, "
                    "analyze data, retrieve knowledge from databases, and even interpret images."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "text": (
                    "nice, do a query on the earth-related topic, give anything about "
                    "the info on the earth form it"
                ),
                "type": "input_text",
            }
        ],
    },
]

KB_MCP_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "mcp_approval_request",
                "tool_call_id": "mcpr_2871b4cd02ff077c006989b97475908190a9e3351213b889b7",
                "name": "knowledge_base_retrieve",
                "arguments": {
                    "request": {
                        "knowledgeAgentIntents": [
                            "Provide general information about the Earth."
                        ]
                    }
                },
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "mcpr_2871b4cd02ff077c006989b97475908190a9e3351213b889b7",
        "role": "tool",
        "content": [
            {"type": "mcp_approval_response", "mcp_approval_response": True}
        ],
    },
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "mcp_2871b4cd02ff077c006989b97c9ee08190a3a68d81a07b8854",
                "name": "knowledge_base_retrieve",
                "arguments": {
                    "request": {
                        "knowledgeAgentIntents": [
                            "Provide general information about the Earth."
                        ]
                    }
                },
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "mcp_2871b4cd02ff077c006989b97c9ee08190a3a68d81a07b8854",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": "Retrieved 11 documents.",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": (
                    "Here's an interesting overview about Earth from space, focusing on nighttime "
                    "images and what they tell us about our planet and humanity."
                ),
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

KB_MCP_TOOL_DEFINITIONS = [
    {
        "name": "knowledge_base_retrieve",
        "type": "function",
        "description": (
            "Knowledge Base Retrieval tool to ask questions. Provide this tool with a list "
            "of knowledge query intents so that the knowledge agent can reason over what "
            "information should be retrieved from its knowledge sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "description": (
                        "Provide this tool with a list of knowledge query intents so that "
                        "the knowledge base can reason over what information should be "
                        "retrieved from its knowledge sources."
                    ),
                    "type": "object",
                    "properties": {
                        "knowledgeAgentIntents": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        }
                    },
                }
            },
            "required": ["request"],
        },
    }
]

# Expected tool_calls after _parse_tools_from_response (MCP approval messages stripped)
KB_MCP_EXPECTED_FLOW_TOOL_CALLS = [
    {
        "type": "tool_call",
        "tool_call_id": "mcp_2871b4cd02ff077c006989b97c9ee08190a3a68d81a07b8854",
        "name": "knowledge_base_retrieve",
        "arguments": {
            "request": {
                "knowledgeAgentIntents": [
                    "Provide general information about the Earth."
                ]
            }
        },
        "tool_result": "Retrieved 11 documents.",
    }
]

# =============================================================================
# 16. MCP Multi-tool
# =============================================================================

MCP_QUERY = [
    {
        "role": "user",
        "content": [
            {
                "text": "can you tell me more about how azure functions work? use MCP",
                "type": "input_text",
            }
        ],
    },
]

MCP_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "mcp_approval_request",
                "tool_call_id": "mcpr_04f33cbf84783da400695a7330ed4c8190b37cc43c1ef54642",
                "name": "microsoft_docs_search",
                "arguments": {"query": "how Azure Functions work"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "mcpr_04f33cbf84783da400695a7330ed4c8190b37cc43c1ef54642",
        "role": "tool",
        "content": [
            {"type": "mcp_approval_response", "mcp_approval_response": True}
        ],
    },
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "mcp_04f33cbf84783da400695a733601048190b28299672103aa49",
                "name": "microsoft_docs_search",
                "arguments": {"query": "how Azure Functions work"},
            }
        ],
    },
    {
        "run_id": "",
        "tool_call_id": "mcp_04f33cbf84783da400695a733601048190b28299672103aa49",
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_result": "Retrieved documentation about Azure Functions.",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": "Azure Functions is a serverless compute service.",
                "type": "output_text",
                "logprobs": [],
            }
        ],
    },
]

MCP_TOOL_DEFINITIONS = [
    {
        "name": "code_interpreter_call",
        "type": "code_interpreter",
        "description": (
            "Use code interpreter to read and interpret information from datasets, "
            "generate code, and create graphs and charts using your data. Supports up to 20 files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Generated code to be executed.",
                }
            },
        },
    },
    {
        "name": "web_search",
        "type": "web_search_preview",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "search_context_size": {
                    "type": "string",
                    "default": "medium",
                },
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
    },
    {
        "name": "microsoft_docs_search",
        "type": "function",
        "description": (
            "Search official Microsoft/Azure documentation to find the most relevant "
            "and trustworthy content for a user's query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "a query or topic about Microsoft/Azure products, services, "
                        "platforms, developer tools, frameworks, or APIs"
                    ),
                    "properties": {},
                    "default": None,
                    "required": [],
                },
                "question": {
                    "type": "string",
                    "description": "this parameter is no longer used, use query instead.",
                    "properties": {},
                    "default": None,
                    "required": [],
                },
            },
            "required": [],
        },
    },
    {
        "name": "microsoft_code_sample_search",
        "type": "function",
        "description": (
            "Search for code snippets and examples in official Microsoft Learn documentation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "a descriptive query, SDK name, method name or code snippet related "
                        "to Microsoft/Azure products"
                    ),
                    "properties": {},
                    "required": [],
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Optional parameter specifying the programming language of code "
                        "snippets to retrieve."
                    ),
                    "properties": {},
                    "default": None,
                    "required": [],
                },
            },
            "required": [],
        },
    },
    {
        "name": "microsoft_docs_fetch",
        "type": "function",
        "description": (
            "Fetch and convert a Microsoft Learn documentation page to markdown format."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the Microsoft documentation page to read",
                    "properties": {},
                    "required": [],
                }
            },
            "required": ["url"],
        },
    },
]

# Expected tool_calls after _parse_tools_from_response (MCP approval messages stripped)
MCP_EXPECTED_FLOW_TOOL_CALLS = [
    {
        "type": "tool_call",
        "tool_call_id": "mcp_04f33cbf84783da400695a733601048190b28299672103aa49",
        "name": "microsoft_docs_search",
        "arguments": {"query": "how Azure Functions work"},
        "tool_result": "Retrieved documentation about Azure Functions.",
    }
]
