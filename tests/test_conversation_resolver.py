from unicorn_chatbot.config import Settings
from unicorn_chatbot.conversation import ConversationResolution, ConversationResolver
from unicorn_chatbot.data import UnicornRepository
from unicorn_chatbot.routing import IntentRouter


class StructuredModel:
    def __init__(self, result):
        self.result = result
        self.schema = None
        self.kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.schema, self.kwargs = schema, kwargs
        return self

    def invoke(self, _messages):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _resolver(repository, result):
    resolver = ConversationResolver(Settings(openai_api_key="test-key"), IntentRouter(repository))
    model = StructuredModel(result)
    resolver._model = lambda: model
    return resolver, model


def test_function_calling_returns_validated_structured_resolution(repository):
    expected = ConversationResolution(
        resolved_query="companies in Bengaluru",
        intent="structured_filter",
        filters={"location": "Bengaluru"},
    )
    resolver, model = _resolver(repository, expected)
    actual = resolver.resolve("What about Bengaluru?", [], [], None)
    assert actual == expected
    assert model.schema is ConversationResolution
    assert model.kwargs == {"method": "function_calling"}
    assert actual.filters == {"location": "Bengaluru"}


def test_structured_clarification_is_preserved(repository):
    expected = ConversationResolution(
        status="clarify",
        intent="collaboration",
        clarification_question="Which collaboration area matters?",
    )
    resolver, _ = _resolver(repository, expected)
    actual = resolver.resolve("Which company is best to collaborate with?", [], [], None)
    assert actual.status == "clarify"
    assert actual.clarification_question == "Which collaboration area matters?"


def test_invalid_structured_output_logs_and_uses_marked_fallback(repository, caplog):
    resolver, _ = _resolver(repository, {"status": "unexpected"})
    actual = resolver.resolve("What about Bengaluru?", [], [], None)
    assert actual.used_fallback
    assert actual.resolved_query == "What about Bengaluru?"
    assert "Conversation resolution failed" in caplog.text


def test_resolver_api_failure_uses_marked_fallback(repository, caplog):
    resolver, _ = _resolver(repository, RuntimeError("transport unavailable"))
    actual = resolver.resolve("What about Bengaluru?", [], [], None)
    assert actual.used_fallback
    assert "RuntimeError" in caplog.text
