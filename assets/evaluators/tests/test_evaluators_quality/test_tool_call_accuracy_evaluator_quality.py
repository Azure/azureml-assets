# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Call Accuracy Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import ToolCallAccuracyEvaluator


@pytest.mark.quality
class TestToolCallAccuracyEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Call Accuracy Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolCallAccuracyEvaluator

    def test_single_call_success(self):
        """Test case: Single call - success.
        
        A single tool call with correct parameters.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Seattle?"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single call - success")
        
        self.assert_pass(result_data)

    def test_single_call_wrong_param1(self):
        """Test case: Single call - wrong param1.
        
        A single tool call with incorrect parameter value.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Seattle?"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single call - wrong param1")
        
        self.assert_fail(result_data)

    def test_single_call_wrong_param2(self):
        """Test case: Single call - wrong param2.
        
        A single tool call with an unexpected parameter.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Get current server time"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
            {
                "name": "get_server_time",
                "description": "Get current server timestamp",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single call - wrong param2")
        
        self.assert_fail(result_data)

    def test_single_call_output_error(self):
        """Test case: Single call - output error.
        
        A single tool call that results in an error.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Delete file report.pdf"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single call - output error")
        
        self.assert_fail(result_data)

    def test_single_call_irrelevant(self):
        """Test case: Single call - irrelevant.
        
        A tool call that doesn't address the user's query.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Seattle?"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single call - irrelevant")
        
        self.assert_fail(result_data)

    def test_missing_calls(self):
        """Test case: Missing calls.
        
        Required tool calls are missing to complete the task.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the distance between Seattle and Portland?"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Missing calls")
        
        self.assert_fail(result_data)

    @pytest.mark.flaky(reruns=3)
    def test_failed_retrial(self):
        """Test case: Failed retrial.
        
        A tool call fails and retry also fails.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Send email to john@example.com"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Failed retrial")
        
        self.assert_fail(result_data)

    def test_succeeded_retrial(self):
        """Test case: Succeeded retrial.
        
        A tool call fails but retry succeeds.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Send email to john@example.com"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Succeeded retrial")
        
        self.assert_pass(result_data)

    def test_duplicate_calls(self):
        """Test case: Duplicate calls.
        
        The same tool call is made multiple times with identical parameters.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Seattle?"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Duplicate calls")
        
        self.assert_pass(result_data)

    def test_unnecessary_incorrect_tool(self):
        """Test case: Unnecessary incorrect tool.
        
        An unnecessary tool call is made alongside the required one.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Seattle?"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
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
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Unnecessary incorrect tool")
        
        self.assert_pass(result_data)
