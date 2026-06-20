from datetime import date, datetime
from email.utils import parsedate_to_datetime
from time import struct_time
from typing import Any


def format_published_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, struct_time):
        return datetime(*value[:6]).date().isoformat()

    value = str(value).strip()
    if not value:
        return None

    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value
