# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Selection Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.tool_selection.evaluator._tool_selection import ToolSelectionEvaluator


@pytest.mark.quality
class TestToolSelectionEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Selection Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolSelectionEvaluator

    def test_single_correct_tool(self):
        """Test case: Single correct tool.
        
        The correct tool is selected for the task.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Send an email to john@example.com with subject 'Meeting'"
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
                            "subject": "Meeting"
                        }
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
                        }
                    },
                    "required": ["to", "subject"]
                }
            },
            {
                "name": "delete_file",
                "description": "Delete a file",
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
        result_data = self._extract_and_print_result(results, "Single correct tool")
        
        self.assert_pass(result_data)

    def test_single_irrelevant_tool(self):
        """Test case: Single irrelevant tool.
        
        An irrelevant tool is selected instead of the correct one.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Send an email to john@example.com"
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
                        "name": "delete_file",
                        "arguments": {
                            "filename": "email.txt"
                        }
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
                        }
                    },
                    "required": ["to"]
                }
            },
            {
                "name": "delete_file",
                "description": "Delete a file",
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
        result_data = self._extract_and_print_result(results, "Single irrelevant tool")
        
        self.assert_fail(result_data)

    def test_missing_calls(self):
        """Test case: Missing calls.
        
        Only part of required tools are called.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Search for iPhone 15 and add it to my cart"
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
                        "name": "product_search",
                        "arguments": {
                            "query": "iPhone 15"
                        }
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "product_search",
                "description": "Search for products",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "add_to_cart",
                "description": "Add product to cart",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string"
                        }
                    },
                    "required": ["product_id"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Missing calls")
        
        self.assert_fail(result_data)

    def test_duplicate_calls(self):
        """Test case: Duplicate calls.
        
        The same tool is called multiple times unnecessarily.
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
                            "to": "john@example.com"
                        }
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_2",
                        "name": "send_email",
                        "arguments": {
                            "to": "john@example.com"
                        }
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
                        }
                    },
                    "required": ["to"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Duplicate calls")
        
        self.assert_pass(result_data)

    def test_necessary_tool_calls_in_conversation_history(self):
        """Test case: Necessary tool calls in conversation history + agent response.
        
        Tool calls are properly selected based on conversation history.
        """
        # TODO: Test fails - evaluator expects fetch_items_in_cart to be called again in response.
        # Reason: "The agent did not call the 'fetch_items_in_cart' tool again, which is essential to retrieve
        # the items in the cart before calculating the total... the agent should have ensured that the necessary
        # data was available for the 'calculate_total' tool."
        # Decision needed: Should the evaluator recognize tools already called in conversation history,
        # or is this expected behavior that response should include all necessary tools?
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Calculate total for items in my cart"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "fetch_items_in_cart",
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
                        "tool_result": "Items: iPhone ($999), Case ($49)"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "You have iPhone and Case in your cart."
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_2",
                        "name": "calculate_total",
                        "arguments": {}
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
                        "tool_call_id": "call_2",
                        "name": "calculate_total",
                        "arguments": {}
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "fetch_items_in_cart",
                "description": "Fetch cart items",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "calculate_total",
                "description": "Calculate cart total",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Necessary tool calls in conversation history")
        
        self.assert_pass(result_data)

    def test_complete_but_excessive(self):
        """Test case: Complete but excessive.
        
        All necessary tools are called but with some unnecessary ones.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Process payment for my order"
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
                        "name": "validate_card",
                        "arguments": {}
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_2",
                        "name": "check_balance",
                        "arguments": {}
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_3",
                        "name": "process_transaction",
                        "arguments": {}
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_4",
                        "name": "send_receipt",
                        "arguments": {}
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_5",
                        "name": "log_transaction",
                        "arguments": {}
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_6",
                        "name": "notify_user",
                        "arguments": {}
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "validate_card",
                "description": "Validate credit card",
                "parameters": {
                    "type": "object"
                }
            },
            {
                "name": "check_balance",
                "description": "Check account balance",
                "parameters": {
                    "type": "object"
                }
            },
            {
                "name": "process_transaction",
                "description": "Process payment",
                "parameters": {
                    "type": "object"
                }
            },
            {
                "name": "send_receipt",
                "description": "Send receipt",
                "parameters": {
                    "type": "object"
                }
            },
            {
                "name": "log_transaction",
                "description": "Log transaction",
                "parameters": {
                    "type": "object"
                }
            },
            {
                "name": "notify_user",
                "description": "Notify user",
                "parameters": {
                    "type": "object"
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Complete but excessive")
        
        self.assert_pass(result_data)
