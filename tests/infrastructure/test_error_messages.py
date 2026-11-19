import httpx
from google.genai import errors

from app.infrastructure.llm.error_messages import plain_error


def api(code, message="x"):
    return errors.APIError(code=code, response_json={"message": message})


def test_daily_quota_says_when_it_resets():
    err = api(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 3h12m53.2s.")
    assert plain_error(err) == "the free daily limit is used up, resets in about 3h 12m"


def test_per_minute_limit():
    assert plain_error(api(429, "per minute")) == "too many requests too fast"


def test_busy_slow_missing_and_key_errors_are_plain():
    assert "busy" in plain_error(api(503))
    assert "too long" in plain_error(api(504))
    assert "does not exist" in plain_error(api(404))
    assert "key" in plain_error(api(403))


def test_network_error():
    assert "internet" in plain_error(httpx.ConnectError("refused"))


def test_messages_hide_the_raw_json():
    text = plain_error(api(503, "{'error': {'code': 503, 'status': 'UNAVAILABLE'}}"))
    assert "{" not in text and "UNAVAILABLE" not in text
