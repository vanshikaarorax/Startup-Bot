"""Runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    dataset_path: Path = Path("data/unicorns.csv")
    embedding_model: str = "text-embedding-3-small"
    vector_index_path: Path = Path("data/vector_index")
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    log_level: str = "INFO"
    max_message_length: int = 2000
    max_session_turns: int = 12
    retrieval_min_score: float = 0.20
    log_query_content: bool = False
    langfuse_enabled: bool = False
    langfuse_capture_content: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        return cls(dataset_path=Path(os.getenv("DATASET_PATH", "data/unicorns.csv")), embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), vector_index_path=Path(os.getenv("VECTOR_INDEX_PATH", "data/vector_index")), openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), openai_api_key=key.strip() if key else None, log_level=os.getenv("LOG_LEVEL", "INFO").upper(), max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", "2000")), max_session_turns=int(os.getenv("MAX_SESSION_TURNS", "12")), retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.20")), log_query_content=os.getenv("LOG_QUERY_CONTENT", "false").lower() == "true", langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true", langfuse_capture_content=os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").lower() == "true", langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None, langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None, langfuse_base_url=os.getenv("LANGFUSE_BASE_URL") or None)
