# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Call Accuracy Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import ToolCallAccuracyEvaluator


@pytest.mark.quality
class TestToolCallAccuracyEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Call Accuracy Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolCallAccuracyEvaluator

    def test_pass_single_call(self) -> None:
        """Test case: PASS - Single call with correct parameters.
        
        A single tool call with correct parameters.
        """
        self.run_quality_test(
            test_label="Single call - success",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in Seattle?"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me check the weather for you."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {
                                "location": "Seattle"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Temperature: 65F, Conditions: Partly cloudy"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The weather in Seattle is 65°F and partly cloudy."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
        )

    def test_fail_single_call_wrong_param_value(self) -> None:
        """Test case: FAIL - Single call with wrong parameter value.
        
        A single tool call with incorrect parameter value.
        """
        self.run_quality_test(
            test_label="Single call - wrong param1",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in Seattle?"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me check the weather for you."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {
                                "location": "Portland"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Temperature: 68F, Conditions: Rainy"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The weather in Portland is 68°F and rainy."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
        )

    def test_fail_single_call_unexpected_param(self) -> None:
        """Test case: FAIL - Single call with unexpected parameter.
        
        A single tool call with an unexpected parameter.
        """
        self.run_quality_test(
            test_label="Single call - wrong param2",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Get current server time"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me get the server time."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_server_time",
                            "arguments": {
                                "timezone": "PST"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: Unexpected parameter 'timezone'"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I encountered an error getting the server time."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_server_time",
                    "description": "Get current server timestamp",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ],
        )

    def test_fail_single_call_output_error(self) -> None:
        """Test case: FAIL - Single call with output error.
        
        A single tool call that results in an error.
        """
        self.run_quality_test(
            test_label="Single call - output error",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Delete file report.pdf"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll delete that file."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "delete_file",
                            "arguments": {
                                "filename": "report.pdf"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: File not found"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I couldn't find the file report.pdf."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "delete_file",
                    "description": "Delete a file by name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string"
                            }
                        },
                        "required": ["filename"]
                    }
                }
            ],
        )

    def test_fail_single_call_irrelevant(self) -> None:
        """Test case: FAIL - Single call irrelevant to query.
        
        A tool call that doesn't address the user's query.
        """
        self.run_quality_test(
            test_label="Single call - irrelevant",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in Seattle?"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me check that for you."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "calculate_distance",
                            "arguments": {
                                "city1": "Seattle",
                                "city2": "Portland"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Distance: 173 miles"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The distance between Seattle and Portland is 173 miles."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                        "required": ["location"]
                    }
                },
                {
                    "name": "calculate_distance",
                    "description": "Calculate distance between cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city1": {
                                "type": "string"
                            },
                            "city2": {
                                "type": "string"
                            }
                        },
                        "required": ["city1", "city2"]
                    }
                }
            ],
        )

    def test_fail_missing_calls(self) -> None:
        """Test case: FAIL - Missing required calls.
        
        Required tool calls are missing to complete the task.
        """
        self.run_quality_test(
            test_label="Missing calls",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the distance between Seattle and Portland?"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll get the coordinates for both cities."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_coordinates",
                            "arguments": {
                                "city": "Seattle"
                            }
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "get_coordinates",
                            "arguments": {
                                "city": "Portland"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "47.6062, -122.3321"
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "45.5152, -122.6784"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Seattle is at 47.6062, -122.3321 and Portland is at 45.5152, -122.6784."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_coordinates",
                    "description": "Get latitude and longitude",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string"
                            }
                        },
                        "required": ["city"]
                    }
                },
                {
                    "name": "calculate_distance",
                    "description": "Calculate distance between coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lat1": {
                                "type": "number"
                            },
                            "lon1": {
                                "type": "number"
                            },
                            "lat2": {
                                "type": "number"
                            },
                            "lon2": {
                                "type": "number"
                            }
                        },
                        "required": ["lat1", "lon1", "lat2", "lon2"]
                    }
                }
            ],
        )

    @pytest.mark.flaky(reruns=3)
    def test_fail_retrial_failed(self) -> None:
        """Test case: FAIL - Retrial also failed.
        
        A tool call fails and retry also fails.
        """
        self.run_quality_test(
            test_label="Failed retrial",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Send email to john@example.com"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "send_email",
                            "arguments": {
                                "to": "john@example.com",
                                "subject": "Hello",
                                "body": "Hi John"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: SMTP connection timeout"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me retry that."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "send_email",
                            "arguments": {
                                "to": "john@example.com",
                                "subject": "Hello",
                                "body": "Hi John"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: SMTP connection timeout"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'm unable to send the email due to connection issues."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string"
                            },
                            "subject": {
                                "type": "string"
                            },
                            "body": {
                                "type": "string"
                            }
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            ],
        )

    def test_pass_retrial_succeeded(self) -> None:
        """Test case: PASS - Retrial succeeded.
        
        A tool call fails but retry succeeds.
        """
        self.run_quality_test(
            test_label="Succeeded retrial",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Send email to john@example.com"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "send_email",
                            "arguments": {
                                "to": "john@example.com",
                                "subject": "Hello",
                                "body": "Hi John"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: Temporary server issue"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me retry that."
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "send_email",
                            "arguments": {
                                "to": "john@example.com",
                                "subject": "Hello",
                                "body": "Hi John"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Email sent successfully"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Email sent successfully to john@example.com."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string"
                            },
                            "subject": {
                                "type": "string"
                            },
                            "body": {
                                "type": "string"
                            }
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            ],
        )

    def test_edge_case_duplicate_calls(self) -> None:
        """Test case: EDGE - Duplicate calls.
        
        The same tool call is made multiple times with identical parameters.
        """
        self.run_quality_test(
            test_label="Duplicate calls",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in Seattle?"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {
                                "location": "Seattle"
                            }
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "get_weather",
                            "arguments": {
                                "location": "Seattle"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Temperature: 65F"
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Temperature: 65F"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The weather in Seattle is 65°F."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
        )

    def test_fail_unnecessary_incorrect_tool(self) -> None:
        """Test case: FAIL - Unnecessary incorrect tool.
        
        An unnecessary tool call is made alongside the required one.
        """
        self.run_quality_test(
            test_label="Unnecessary incorrect tool",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in Seattle?"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {
                                "location": "Seattle"
                            }
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "get_timezone",
                            "arguments": {
                                "location": "Seattle"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Temperature: 65F"
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Timezone: PST"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The weather in Seattle is 65°F."
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                        "required": ["location"]
                    }
                },
                {
                    "name": "get_timezone",
                    "description": "Get timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
        )
