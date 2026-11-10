from fastapi import Header, Request

from app.container import Container
from app.domain.entities.user import CurrentUser


def get_container(request: Request) -> Container:
    """Retrieves the application container from the running FastAPI app state."""
    return request.app.state.container


def get_current_user(
    authorization: str | None = Header(None, description="Optional Bearer token"),
) -> CurrentUser:
    """Resolves caller identity.

    In production, this extracts the subject from a verified JWT token.
    In local development / tests without a token, defaults to a standard student identity.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        # Stub/verifier hook: in future or when JWT is active, verify HS256 token
        return CurrentUser(email=f"user_{token[:8]}@shine.lk", role="student", user_id=1)

    return CurrentUser(email="student@shine.lk", role="student", user_id=1)

