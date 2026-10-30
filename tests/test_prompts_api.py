from fastapi.testclient import TestClient

from unicorn_chatbot.api import ChatbotService, create_app
from unicorn_chatbot.config import Settings
from unicorn_chatbot.prompts import SYSTEM_MESSAGE, build_chat_prompt
from unicorn_chatbot.service import ChatbotService


def test_prompt_has_grounding_instruction():
    prompt = build_chat_prompt("What is Alpha Pay?", [{"company_name": "Alpha Pay"}])
    assert "Never use outside knowledge" in SYSTEM_MESSAGE
    assert "Alpha Pay" in str(prompt.format_messages())


def test_api_validates_message(repository):
    class StubService:
        seen_session_id = None
        def answer(self, message, session_id):
            self.seen_session_id = session_id
            return ("grounded", [{"company_name": "Alpha Pay"}], "structured")
    stub = StubService()
    app = create_app(Settings(), stub)
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422
        response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json()["answer"] == "grounded"
    assert response.json()["session_id"] == stub.seen_session_id

