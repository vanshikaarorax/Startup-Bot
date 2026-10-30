"""Backward-compatible evidence-grounded answer prompt exports."""
from .generation.prompts import SYSTEM_MESSAGE, build_chat_prompt

__all__ = ["SYSTEM_MESSAGE", "build_chat_prompt"]
