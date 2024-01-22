from abc import ABC, abstractmethod


class RequestModifier(ABC):
    @abstractmethod
    def modify(self, request_obj: any) -> any:
        pass

class RequestModificationException(Exception):
    def __init__(self, message: str = "An exception was thrown while attempting to apply a RequestModifier.") -> None:
        super().__init__(message)