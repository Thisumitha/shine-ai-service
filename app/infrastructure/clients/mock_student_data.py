import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.domain.ports.student_data import StudentDataError, StudentDataProvider

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "sample" / "student_data.json"


class MockStudentDataClient(StudentDataProvider):
    """Serves student records from a JSON file, so the MCP tools work without Spring.

    Timetable rows use `dayOffset` (days from today) so the demo never goes stale.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        file = Path(path) if path else DEFAULT_PATH
        try:
            self._data: dict[str, dict[str, list[dict[str, Any]]]] = json.loads(
                file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise StudentDataError(f"Could not read mock student data from {file}") from exc

    def timetable(self, student_email: str) -> list[dict[str, Any]]:
        today = date.today()
        slots = []
        for row in self._rows(student_email, "timetable"):
            row = dict(row)
            if "dayOffset" in row:
                row["date"] = (today + timedelta(days=int(row.pop("dayOffset")))).isoformat()
            slots.append(row)
        return slots

    def invoices(self, student_email: str) -> list[dict[str, Any]]:
        return self._rows(student_email, "invoices")

    def attendance(self, student_email: str) -> list[dict[str, Any]]:
        return self._rows(student_email, "attendance")

    def _rows(self, student_email: str, key: str) -> list[dict[str, Any]]:
        student = self._data.get(student_email.strip().lower())
        if student is None:
            raise StudentDataError(f"No mock record for {student_email}")
        return list(student.get(key, []))
