"""Validation and normalization for untrusted HTTP input."""
from __future__ import annotations

import re
import uuid


class InputError(ValueError):
    """A safe client-correctable validation error."""


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def sanitize_message(value: str, max_length: int) -> str:
    message = " ".join(value.strip().split())
    if not message:
        raise InputError("message must not be empty")
    if len(message) > max_length:
        raise InputError(f"message exceeds the {max_length}-character limit")
    return message


def validate_session_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    if not SESSION_ID_PATTERN.fullmatch(value):
        raise InputError("session_id must be 1-64 characters using letters, numbers, hyphens, or underscores")
    return value
