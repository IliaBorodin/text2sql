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


class ValidationError(Text2SQLError):
    """Raised when user input or intermediate data fails validation."""


class SchemaRetrievalError(Text2SQLError):
    """Raised when database schema metadata cannot be loaded."""


class PromptBuildError(Text2SQLError):
    """Raised when the LLM prompt cannot be assembled."""


__all__ = [
    "DatabaseError",
    "LLMError",
    "PromptBuildError",
    "SchemaRetrievalError",
    "Text2SQLError",
    "ValidationError",
]
