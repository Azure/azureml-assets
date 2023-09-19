from .modifiers.request_modifier import RequestModifier
from ..utils import logging_utils as lu

class InputTransformer():
    def __init__(self, modifiers: "list[RequestModifier]") -> None:
        self.__modifiers = modifiers
    
    def apply_modifications(self, request_obj: any):
        if self.__modifiers and len(self.__modifiers) > 0:
            lu.get_logger().info(f"Applying InputTransformer modifications")
            for modifier in self.__modifiers:
                request_obj = modifier.modify(request_obj)
        else:
            lu.get_logger().info("No InputTransformer modifications configured")

        return request_obj
