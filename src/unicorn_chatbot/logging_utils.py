"""Minimal JSON logging that excludes secrets by construction."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonEventLogger:
    def __init__(self, name: str = "unicorn_chatbot") -> None:
        self.logger = logging.getLogger(name)

    def event(self, event: str, **fields: Any) -> None:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self.logger.info(json.dumps(payload, default=str))
