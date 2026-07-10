from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeckVisionError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message} {self.details}"

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}
