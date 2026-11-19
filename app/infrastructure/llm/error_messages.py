"""Turn a raw Gemini / network exception into one short sentence anyone can read.

The full technical text is still logged at DEBUG level for developers.
"""
import re

from google.genai import errors

from app.infrastructure.network_errors import is_network_error


def _reset_hint(text: str) -> str:
    match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?", text)
    if not match:
        return ""
    hours, minutes = int(match.group(1) or 0), int(match.group(2) or 0)
    if hours:
        return f", resets in about {hours}h {minutes}m" if minutes else f", resets in about {hours}h"
    return f", resets in about {minutes} min" if minutes else ""


class LLMUnavailableError(RuntimeError):
    """A model backend (Ollama, Claude Code ...) cannot be used right now. The message is already plain."""


def is_service_error(err: BaseException) -> bool:
    """A Google API or network failure (as opposed to a bug in our own code)."""
    return isinstance(err, errors.APIError) or is_network_error(err)


def is_daily_quota(err: BaseException) -> bool:
    """A per-day quota stays used up for hours, so retrying it only wastes requests."""
    return isinstance(err, errors.APIError) and getattr(err, "code", None) == 429 and "PerDay" in str(err)


def plain_error(err: BaseException) -> str:
    """e.g. 'the free daily limit is used up, resets in about 3h 12m'."""
    if isinstance(err, LLMUnavailableError):
        return str(err)
    if isinstance(err, errors.APIError):
        code = getattr(err, "code", None)
        text = str(err)
        if code == 429:
            if "PerDay" in text:
                return "the free daily limit is used up" + _reset_hint(text)
            return "too many requests too fast"
        if code in (500, 502, 503):
            return "Google's servers are busy right now"
        if code == 504:
            return "Google took too long to reply"
        if code == 404:
            return "this model does not exist for your key"
        if code in (401, 403):
            return "the API key was rejected"
        if code == 400:
            return "the request was not accepted"
        return f"Google returned an error ({code})"
    if is_network_error(err):
        return "no connection to Google (check your internet)"
    return "something went wrong"
