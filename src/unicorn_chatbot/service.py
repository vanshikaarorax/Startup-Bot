"""Backward-compatible ChatbotService import."""
from .conversation.schemas import ConversationResolution
from .orchestration.service import ChatbotService

__all__ = ["ChatbotService", "ConversationResolution"]
