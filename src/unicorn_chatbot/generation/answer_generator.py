"""Grounded answer generation only; no routing or retrieval decisions."""
from __future__ import annotations
import logging
from typing import Any
from langchain_openai import ChatOpenAI
from ..core.config import Settings
from .prompts import build_chat_prompt

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None: self.settings = settings
    def generate(self, query: str, records: list[dict[str, Any]], history: list[Any]) -> str:
        if not records: return "I could not find matching information in the supplied dataset."
        if not self.settings.openai_api_key: return "Matching dataset records were found, but response generation is unavailable until OPENAI_API_KEY is configured."
        try:
            response = (build_chat_prompt(query, records, history) | ChatOpenAI(model=self.settings.openai_model, api_key=self.settings.openai_api_key, temperature=0)).invoke({})
            answer = str(response.content).strip()
            if not answer: raise ValueError("The answer model returned an empty response.")
            return answer
        except Exception:
            logger.exception("Grounded answer generation failed.")
            return "I found relevant dataset records but could not generate a response. Please try again."
