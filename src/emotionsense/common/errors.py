"""Typed exception hierarchy mapping cleanly to HTTP error envelopes.

Interior code raises these; the API layer converts them to the standard error envelope
(see docs/design/04-api-specification.md). Never swallow exceptions — catch narrow, add
context, re-raise as an ``AppError``.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base application error. Carries a machine code + client-safe message."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_envelope(self, correlation_id: str | None = None) -> dict[str, Any]:
        """Serialize to the API error envelope shape."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "correlation_id": correlation_id,
            }
        }


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    http_status = 400


class UnauthenticatedError(AppError):
    code = "UNAUTHENTICATED"
    http_status = 401


class ForbiddenError(AppError):
    code = "FORBIDDEN"
    http_status = 403


class NotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404


class ConflictError(AppError):
    code = "CONFLICT"
    http_status = 409


class PayloadTooLargeError(AppError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class UnsupportedMediaTypeError(AppError):
    code = "UNSUPPORTED_MEDIA_TYPE"
    http_status = 415


class ModelUnavailableError(AppError):
    """Requested model version could not be loaded — degrade gracefully (NFR-9)."""

    code = "MODEL_UNAVAILABLE"
    http_status = 503
