"""Actionable HTTP errors without leaking request inputs."""


class ApiError(Exception):
    """Transport failure; existing NexuML errors remain authoritative."""

    def __init__(self, status: int, code: str, message: str, fields: list | None = None):
        self.status = status
        self.code = code
        self.message = message
        self.fields = fields or []
