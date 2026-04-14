"""Custom project exceptions."""


class Text2SQLError(Exception):
    """Base project exception."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class LLMError(Text2SQLError):
    """Raised when LLM interaction fails."""


class DatabaseError(Text2SQLError):
    """Raised when SQL execution fails."""


__all__ = ["DatabaseError", "LLMError", "Text2SQLError"]
