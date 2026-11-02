from dataclasses import dataclass


@dataclass(frozen=True)
class CurrentUser:
    """The caller, taken from the verified login token (never from the request body)."""

    email: str
    role: str
    user_id: int | None = None
