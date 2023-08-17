from datetime import datetime
from dateutil import parser

def format_date_string(format: str, date_to_format: str) -> datetime:
    parsed_date = parser.parse(date_to_format)
    return datetime.strptime(
        str(parsed_date.strftime(format)), format
    )
