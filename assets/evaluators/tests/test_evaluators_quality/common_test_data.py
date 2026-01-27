# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Common test data, tool definitions, and constants for quality tests.

This module provides shared resources to ensure consistency across quality test files.
"""

from typing import Dict, Any, List


# =============================================================================
# COMMON TOOL DEFINITIONS
# =============================================================================

# Type alias for tool definition dictionaries
ToolDefinition = Dict[str, Any]


class ToolDefinitions:
    """Reusable tool definitions for quality tests."""

    # Email Tool
    SEND_EMAIL: ToolDefinition = {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"}
            },
            "required": ["to", "subject"]
        }
    }

    SEND_EMAIL_BASIC: ToolDefinition = {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"}
            },
            "required": ["to", "subject"]
        }
    }

    # Calculator Tool
    CALCULATE: ToolDefinition = {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Math operation (add, subtract, multiply, divide)"},
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"}
            },
            "required": ["operation", "a", "b"]
        }
    }

    # File Operations
    DELETE_FILE: ToolDefinition = {
        "name": "delete_file",
        "description": "Delete a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the file to delete"}
            },
            "required": ["filename"]
        }
    }

    GET_FILES_IN_FOLDER: ToolDefinition = {
        "name": "GetFilesInFolder",
        "description": "List all files in a folder",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Folder path"}
            },
            "required": ["path"]
        }
    }

    # Product Search
    PRODUCT_SEARCH: ToolDefinition = {
        "name": "product_search",
        "description": "Search for products",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }

    ADD_TO_CART: ToolDefinition = {
        "name": "add_to_cart",
        "description": "Add item to shopping cart",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product ID to add"}
            },
            "required": ["product_id"]
        }
    }

    # Weather Tool
    GET_WEATHER: ToolDefinition = {
        "name": "GetWeather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Location to get weather for"},
                "units": {"type": "string", "description": "Temperature units (celsius, fahrenheit)"}
            },
            "required": ["location"]
        }
    }

    # Time Tools
    GET_CURRENT_TIME: ToolDefinition = {
        "name": "GetCurrentTime",
        "description": "Get current time",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    GET_SERVER_TIME: ToolDefinition = {
        "name": "get_server_time",
        "description": "Get current server time",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone (e.g., PST, EST, UTC)"}
            },
            "required": ["timezone"]
        }
    }

    # Booking Tools
    BOOK_FLIGHT: ToolDefinition = {
        "name": "book_flight",
        "description": "Book a flight",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Departure city"},
                "to": {"type": "string", "description": "Destination city"},
                "departure_date": {"type": "string", "description": "Departure date"},
                "return_date": {"type": "string", "description": "Return date (for round-trip)"},
                "trip_type": {"type": "string", "description": "Trip type (one-way, round-trip)"}
            },
            "required": ["from", "to", "departure_date"]
        }
    }

    BOOK_APPOINTMENT: ToolDefinition = {
        "name": "book_appointment",
        "description": "Book an appointment",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    }

    # User Management
    GET_USER: ToolDefinition = {
        "name": "get_user",
        "description": "Get user by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "User ID"}
            },
            "required": ["user_id"]
        }
    }

    # Order Management
    GET_ORDER: ToolDefinition = {
        "name": "get_order",
        "description": "Get order details",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"}
            },
            "required": ["order_id"]
        }
    }

    GET_RECENT_ORDERS: ToolDefinition = {
        "name": "get_recent_orders",
        "description": "Get recent orders",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    GET_ORDER_STATUS: ToolDefinition = {
        "name": "get_order_status",
        "description": "Get order status",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"}
            },
            "required": ["order_id"]
        }
    }

    # Location Tools
    GET_COORDINATES: ToolDefinition = {
        "name": "get_coordinates",
        "description": "Get coordinates for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }

    CALCULATE_DISTANCE: ToolDefinition = {
        "name": "calculate_distance",
        "description": "Calculate distance between two points",
        "parameters": {
            "type": "object",
            "properties": {
                "lat1": {"type": "number", "description": "Latitude of first point"},
                "lon1": {"type": "number", "description": "Longitude of first point"},
                "lat2": {"type": "number", "description": "Latitude of second point"},
                "lon2": {"type": "number", "description": "Longitude of second point"}
            },
            "required": ["lat1", "lon1", "lat2", "lon2"]
        }
    }

    # Cart Tools
    FETCH_ITEMS_IN_CART: ToolDefinition = {
        "name": "fetch_items_in_cart",
        "description": "Fetch cart items",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    CALCULATE_TOTAL: ToolDefinition = {
        "name": "calculate_total",
        "description": "Calculate cart total",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    # Month Names Tool
    GET_MONTH_NAMES_LIST: ToolDefinition = {
        "name": "GetMonthNamesList",
        "description": "Get list of all month names",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    # Payment Processing Tools
    VALIDATE_CARD: ToolDefinition = {
        "name": "validate_card",
        "description": "Validate credit card",
        "parameters": {"type": "object"}
    }

    CHECK_BALANCE: ToolDefinition = {
        "name": "check_balance",
        "description": "Check account balance",
        "parameters": {"type": "object"}
    }

    PROCESS_TRANSACTION: ToolDefinition = {
        "name": "process_transaction",
        "description": "Process payment",
        "parameters": {"type": "object"}
    }

    SEND_RECEIPT: ToolDefinition = {
        "name": "send_receipt",
        "description": "Send receipt",
        "parameters": {"type": "object"}
    }

    LOG_TRANSACTION: ToolDefinition = {
        "name": "log_transaction",
        "description": "Log transaction",
        "parameters": {"type": "object"}
    }

    NOTIFY_USER: ToolDefinition = {
        "name": "notify_user",
        "description": "Notify user",
        "parameters": {"type": "object"}
    }


# =============================================================================
# COMMON TOOL DEFINITION SETS
# =============================================================================

class ToolDefinitionSets:
    """Pre-configured sets of tool definitions for common test scenarios."""

    EMAIL_AND_FILE: List[ToolDefinition] = [
        ToolDefinitions.SEND_EMAIL_BASIC,
        ToolDefinitions.DELETE_FILE
    ]

    SHOPPING: List[ToolDefinition] = [
        ToolDefinitions.PRODUCT_SEARCH,
        ToolDefinitions.ADD_TO_CART
    ]

    FLIGHT_BOOKING: List[ToolDefinition] = [
        ToolDefinitions.BOOK_FLIGHT,
        ToolDefinitions.SEND_EMAIL
    ]

    COORDINATES_AND_DISTANCE: List[ToolDefinition] = [
        ToolDefinitions.GET_COORDINATES,
        ToolDefinitions.CALCULATE_DISTANCE
    ]

    CART_OPERATIONS: List[ToolDefinition] = [
        ToolDefinitions.FETCH_ITEMS_IN_CART,
        ToolDefinitions.CALCULATE_TOTAL
    ]

    PAYMENT_PROCESSING: List[ToolDefinition] = [
        ToolDefinitions.VALIDATE_CARD,
        ToolDefinitions.CHECK_BALANCE,
        ToolDefinitions.PROCESS_TRANSACTION,
        ToolDefinitions.SEND_RECEIPT,
        ToolDefinitions.LOG_TRANSACTION,
        ToolDefinitions.NOTIFY_USER
    ]


# =============================================================================
# COMMON RESPONSE TEXTS
# =============================================================================

class ResponseTexts:
    """Common response texts used across tests."""

    # Climate Change Analysis (highly coherent)
    CLIMATE_CHANGE_ANALYSIS = (
        "Climate change significantly affects the economies of coastal cities through "
        "rising sea levels, increased flooding, and more intense storms. These environmental "
        "changes can damage infrastructure, disrupt businesses, and lead to costly repairs. "
        "For instance, frequent flooding can hinder transportation and commerce, while the "
        "threat of severe weather may deter investment and tourism. Consequently, cities may "
        "face increased expenses for disaster preparedness and mitigation efforts, straining "
        "municipal budgets and impacting economic growth. Furthermore, property values in "
        "vulnerable areas may decline, reducing tax revenues and affecting homeowners' wealth. "
        "In response, many coastal cities are investing in resilient infrastructure and "
        "exploring innovative solutions to protect their economies for the future."
    )

    # Water Cycle Explanation (coherent)
    WATER_CYCLE_EXPLANATION = (
        "The water cycle is the continuous movement of water on Earth through several "
        "key processes. First, water evaporates from oceans, lakes, and rivers when heated "
        "by the sun. This water vapor rises into the atmosphere where it cools and condenses "
        "to form clouds. When the water droplets in clouds become heavy enough, they fall "
        "back to Earth as precipitation, such as rain or snow. This water then collects in "
        "bodies of water or seeps into the ground, eventually making its way back to the "
        "oceans to begin the cycle again. This continuous process is essential for "
        "distributing water resources across the planet."
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_user_message(text: str) -> Dict[str, Any]:
    """Create a standard user message format."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}]
    }


def create_assistant_text_message(text: str) -> Dict[str, Any]:
    """Create a standard assistant text message format."""
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}]
    }


def create_tool_call(
    tool_call_id: str,
    name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a tool call content item."""
    return {
        "type": "tool_call",
        "tool_call_id": tool_call_id,
        "name": name,
        "arguments": arguments
    }


def create_assistant_tool_call_message(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create an assistant message with tool calls."""
    return {
        "role": "assistant",
        "content": tool_calls
    }


def create_tool_result_message(
    tool_call_id: str,
    result: Any
) -> Dict[str, Any]:
    """Create a tool result message."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [{"type": "tool_result", "tool_result": result}]
    }
