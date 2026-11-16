import logging
from typing import Any

import httpx

from app.domain.ports.student_data import StudentDataError, StudentDataProvider

logger = logging.getLogger(__name__)


class SpringStudentDataClient(StudentDataProvider):
    """Reads student data from the Spring backend's /internal/student/* endpoints.

    Authenticates with the shared X-Internal-Secret header. The student email is a
    query parameter that Spring verified before calling this service.
    """

    def __init__(
        self,
        base_url: str,
        internal_secret: str,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._secret = internal_secret
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def timetable(self, student_email: str) -> list[dict[str, Any]]:
        return self._get("timetable", student_email)

    def invoices(self, student_email: str) -> list[dict[str, Any]]:
        return self._get("invoices", student_email)

    def attendance(self, student_email: str) -> list[dict[str, Any]]:
        return self._get("attendance", student_email)

    def _get(self, resource: str, student_email: str) -> list[dict[str, Any]]:
        url = f"{self._base_url}/internal/student/{resource}"
        try:
            response = self._client.get(
                url,
                params={"studentEmail": student_email},
                headers={"X-Internal-Secret": self._secret},
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("Student data request to %s failed: %s", url, exc)
            raise StudentDataError(f"Could not load {resource}") from exc

        if not isinstance(data, list):
            raise StudentDataError(f"Unexpected {resource} response shape")
        return data
