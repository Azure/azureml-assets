# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Row Validation Context"""

from dataclasses import dataclass
from typing import Optional

from row_validators.input_row import InputRow


@dataclass
class RowValidationContext:
    """Row Validation Context"""

    raw_input_row: str
    parsed_input_row: InputRow
    line_number: Optional[int]

    def __init__(self, raw_input_row: str, line_number: Optional[int] = None):
        self.raw_input_row = raw_input_row
        self.parsed_input_row = None
        self.line_number = line_number
