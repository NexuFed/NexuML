"""Actionable HTTP errors without leaking request inputs."""


def validation_fields(error) -> list[dict]:
    """Project validation errors onto public config fields without inputs/context.

    Returns:
        Field locations, messages and types without internal training-union labels.
    """
    fields = []
    for item in error.errors(include_input=False, include_context=False):
        loc = list(item["loc"])
        # Union branch names are not fields in the portable training document.
        if (
            len(loc) > 1
            and loc[0] == "training"
            and loc[1]
            in {
                "TrainingSpec",
                "ReinforcementLearningSpec",
            }
        ):
            del loc[1]
        fields.append({"loc": loc, "message": item["msg"], "type": item["type"]})
    return fields


class ApiError(Exception):
    """Transport failure; existing NexuML errors remain authoritative."""

    def __init__(self, status: int, code: str, message: str, fields: list | None = None):
        self.status = status
        self.code = code
        self.message = message
        self.fields = fields or []
