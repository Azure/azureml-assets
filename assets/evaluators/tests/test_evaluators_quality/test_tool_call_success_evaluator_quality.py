# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Call Success Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.tool_call_success.evaluator._tool_call_success import ToolCallSuccessEvaluator


@pytest.mark.quality
class TestToolCallSuccessEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Call Success Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolCallSuccessEvaluator

    def test_single_call_success(self):
        """Test case: Single-call-success.
        
        A single tool call that executes successfully.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's 25 * 4?"
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
                        "name": "calculate",
                        "arguments": {
                            "operation": "multiply",
                            "a": 25,
                            "b": 4
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
                        "tool_result": "100"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "25 * 4 = 100"
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string"
                        },
                        "a": {
                            "type": "number"
                        },
                        "b": {
                            "type": "number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single-call-success")
        
        self.assert_pass(result_data)

    def test_single_call_fail(self):
        """Test case: Single-call-fail.
        
        A single tool call that fails with an error.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's 25 / 0?"
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
                        "name": "calculate",
                        "arguments": {
                            "operation": "divide",
                            "a": 25,
                            "b": 0
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
                        "tool_result": "Error: Division by zero"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I can't divide by zero."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string"
                        },
                        "a": {
                            "type": "number"
                        },
                        "b": {
                            "type": "number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Single-call-fail")
        
        self.assert_fail(result_data)

    def test_multi_call_single_fail(self):
        """Test case: Multi-call Single Fail.
        
        Multiple tool calls where one fails but others succeed.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Calculate 10+5, 20/0, and 3*7"
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
                        "name": "calculate",
                        "arguments": {
                            "operation": "add",
                            "a": 10,
                            "b": 5
                        }
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_2",
                        "name": "calculate",
                        "arguments": {
                            "operation": "divide",
                            "a": 20,
                            "b": 0
                        }
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_3",
                        "name": "calculate",
                        "arguments": {
                            "operation": "multiply",
                            "a": 3,
                            "b": 7
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
                        "tool_result": "15"
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": "Error: Division by zero"
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": "21"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "10+5=15, cannot divide by zero, 3*7=21"
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string"
                        },
                        "a": {
                            "type": "number"
                        },
                        "b": {
                            "type": "number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Multi-call Single Fail")
        
        self.assert_fail(result_data)

    def test_legal_empty_response(self):
        """Test case: Legal-EmptyResponse.
        
        A tool call returns an empty result, which is valid (folder has no files).
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "List files in /empty folder"
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
                        "name": "GetFilesInFolder",
                        "arguments": {
                            "path": "/empty"
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
                        "tool_result": []
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The folder is empty."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetFilesInFolder",
                "description": "List all files in a folder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string"
                        }
                    },
                    "required": ["path"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Legal-EmptyResponse")
        
        self.assert_pass(result_data)

    def test_empty_string(self):
        """Test case: Empty-string.
        
        A tool returns an empty string which indicates an error.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Get current time"
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
                        "name": "GetTimeNow",
                        "arguments": {}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": ""
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Unable to get current time."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetTimeNow",
                "description": "Get current server time",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Empty-string")
        
        self.assert_fail(result_data)

    def test_empty_json_object(self):
        """Test case: Empty-Json-Object.
        
        A tool returns a JSON object with all empty fields.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Get today's date"
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
                        "name": "GetDate",
                        "arguments": {}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": {
                            "Day": "",
                            "Month": "",
                            "Year": ""
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Unable to retrieve the date."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetDate",
                "description": "Get current date",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Empty-Json-Object")
        
        self.assert_fail(result_data)

    def test_empty_array(self):
        """Test case: Empty-Array.
        
        A tool that should return data returns an empty array instead.
        """
        # TODO: Test fails - evaluator passes empty array as valid tool response.
        # Reason: "The tool call to GetMonthNamesList returned an empty list, which is acceptable
        # as it does not indicate a technical error based on the tool's definition."
        # Decision needed: Should empty array from a tool that should return data be considered failure,
        # or is empty response technically a successful tool execution?
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "List all month names"
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
                        "name": "GetMonthNamesList",
                        "arguments": {}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": []
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Unable to retrieve month names."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetMonthNamesList",
                "description": "Get list of all month names",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Empty-Array")
        
        self.assert_fail(result_data)

    def test_exception_object(self):
        """Test case: Exception-Object.
        
        A tool returns an error object with error code and description.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Query the database"
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
                        "name": "QueryDatabase",
                        "arguments": {
                            "query": "SELECT * FROM users"
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
                        "tool_result": {
                            "ErrorCode": "0858",
                            "ErrorDesc": "Failed to access database. Server is down"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Database server is currently down."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "QueryDatabase",
                "description": "Execute database query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Exception-Object")
        
        self.assert_fail(result_data)

    def test_exception_as_string(self):
        """Test case: Exception-as-string.
        
        A tool returns an error message as a plain string.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Connect to server"
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
                        "name": "ConnectServer",
                        "arguments": {
                            "server": "api.example.com"
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
                        "tool_result": "The connection was reused by the server. Access without SSL is not allowed"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Connection failed: SSL is required."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "ConnectServer",
                "description": "Connect to a server",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server": {
                            "type": "string"
                        }
                    },
                    "required": ["server"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Exception-as-string")
        
        self.assert_fail(result_data)

    def test_failure_status_code(self):
        """Test case: Failure-Status-Code.
        
        A tool returns only an HTTP error status code.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Fetch user data"
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
                        "name": "FetchData",
                        "arguments": {
                            "endpoint": "/api/users"
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
                        "tool_result": "404"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "User data not found."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "FetchData",
                "description": "Fetch data from API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string"
                        }
                    },
                    "required": ["endpoint"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Failure-Status-Code")
        
        self.assert_fail(result_data)

    def test_json_object_with_exception_string_field(self):
        """Test case: Json-Object-With-Exception-String-Field.
        
        A tool returns a JSON object with an error field populated.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Get person information"
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
                        "name": "GetPersonInformation",
                        "arguments": {
                            "person_id": "123"
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
                        "tool_result": {
                            "Name": "",
                            "Age": "",
                            "LastError": "Person information could not be accessed due to insufficient permission"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Access denied: insufficient permissions."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetPersonInformation",
                "description": "Get person details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person_id": {
                            "type": "string"
                        }
                    },
                    "required": ["person_id"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Json-Object-With-Exception-String-Field")
        
        self.assert_fail(result_data)

    def test_null_result(self):
        """Test case: Null-Result.
        
        A tool returns null as its result.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Get current time"
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
                        "name": "GetTimeNow",
                        "arguments": {}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": None
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Unable to get current time."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetTimeNow",
                "description": "Get current server time",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Null-Result")
        
        self.assert_fail(result_data)

    def test_timeout(self):
        """Test case: Time-out.
        
        A tool call times out.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Run long query"
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
                        "name": "ExecuteQuery",
                        "arguments": {
                            "query": "SELECT * FROM huge_table"
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
                        "tool_result": "Connection timeout"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Query timed out."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "ExecuteQuery",
                "description": "Execute database query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Time-out")
        
        self.assert_fail(result_data)

    def test_retries(self):
        """Test case: Retries.
        
        A tool call fails then is retried successfully with same parameters.
        """
        # TODO: Test fails - evaluator fails despite successful retry.
        # Reason: "The first tool call to GetWeather resulted in a temporary network issue, indicating a failure.
        # The second call succeeded with valid weather information. {'failed_tools': 'GetWeather', 'success': False}"
        # Decision needed: Should a successful retry after initial failure pass,
        # or does any failed tool call in the conversation mean overall failure?
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Get weather data"
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
                        "name": "GetWeather",
                        "arguments": {
                            "location": "Seattle",
                            "units": "fahrenheit"
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
                        "tool_result": "Error: Temporary network issue"
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
                        "name": "GetWeather",
                        "arguments": {
                            "location": "Seattle",
                            "units": "fahrenheit"
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
                        "tool_result": "Temperature: 65F, Conditions: Partly cloudy"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Weather in Seattle: 65°F, partly cloudy"
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "GetWeather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string"
                        },
                        "units": {
                            "type": "string"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Retries")
        
        self.assert_pass(result_data)
