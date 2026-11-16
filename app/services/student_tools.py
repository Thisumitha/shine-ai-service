"""Read-only student-data tools shared by the MCP server and the chat agent.

The student email is never a tool argument: callers pass it separately, after Spring
has verified it, so the LLM cannot ask for another student's data.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from app.domain.ports.student_data import StudentDataProvider

MAX_ITEMS = 25
CURRENCY = "LKR"  # every Shine amount is in Sri Lankan rupees

# Attendance rules shared with the dashboard and admin report (see CLAUDE.md).
AT_RISK_BELOW = 50.0
EXCELLENT_ABOVE = 80.0

# Fields from Spring that must never reach the LLM (payment links, slip files).
_HIDDEN_INVOICE_FIELDS = {"paymentLink", "bankSlipUrl", "items"}


def attendance_status(percentage: float, total_classes: int) -> str:
    if total_classes <= 0:
        return "No Sessions Yet"
    if percentage < AT_RISK_BELOW:
        return "Student at risk - Future attendance Compulsory"
    if percentage <= EXCELLENT_ABOVE:
        return "Consistent Participation Required"
    return "Excellent"


@dataclass(frozen=True)
class ToolSpec:
    """Describes one tool: what the LLM sees (name, description, params) and how to run it."""

    name: str
    description: str
    parameters: dict[str, str]
    run: Callable[[str, dict[str, Any]], Any]


class StudentToolService:
    """Implements the student-data tools on top of a StudentDataProvider."""

    def __init__(self, provider: StudentDataProvider) -> None:
        self._provider = provider
        self.tools: dict[str, ToolSpec] = {
            spec.name: spec
            for spec in (
                ToolSpec(
                    "get_timetable",
                    "The student's classes and exams. Optional date (YYYY-MM-DD) limits to one day; "
                    "omit it for upcoming items.",
                    {"date": "optional YYYY-MM-DD"},
                    lambda email, args: self.get_timetable(email, args.get("date")),
                ),
                ToolSpec(
                    "get_invoices",
                    "The student's payment invoices. Optional status PENDING or PAID.",
                    {"status": "optional PENDING or PAID"},
                    lambda email, args: self.get_invoices(email, args.get("status")),
                ),
                ToolSpec(
                    "get_attendance",
                    "The student's attendance percentage per course with a status label.",
                    {},
                    lambda email, args: self.get_attendance(email),
                ),
            )
        }

    def call(self, name: str, student_email: str, args: dict[str, Any] | None = None) -> Any:
        spec = self.tools.get(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")
        return spec.run(student_email, args or {})

    def get_timetable(self, student_email: str, day: str | None = None) -> list[dict[str, Any]]:
        slots = self._provider.timetable(student_email)
        today = date.today().isoformat()
        wanted = day or None
        picked = [
            s for s in slots
            if (wanted and str(s.get("date")) == wanted) or (not wanted and str(s.get("date")) >= today)
        ]
        picked.sort(key=lambda s: (str(s.get("date")), str(s.get("startTime"))))
        return [
            {
                "name": s.get("className"),
                "type": s.get("eventType"),
                "date": s.get("date"),
                "start": s.get("startTime"),
                "end": s.get("endTime"),
                "status": s.get("status"),
            }
            for s in picked[:MAX_ITEMS]
        ]

    def get_invoices(self, student_email: str, status: str | None = None) -> list[dict[str, Any]]:
        wanted = status.strip().upper() if status else None
        rows = [
            {k: v for k, v in inv.items() if k not in _HIDDEN_INVOICE_FIELDS}
            for inv in self._provider.invoices(student_email)
        ]
        if wanted:
            rows = [r for r in rows if str(r.get("paymentStatus", "")).upper() == wanted]
        return [
            {
                "invoiceNumber": r.get("invoiceNumber"),
                "amount": r.get("totalAmount"),
                "currency": CURRENCY,
                "status": r.get("paymentStatus"),
                "dueDate": r.get("dueDate"),
                "paidAt": r.get("paidAt"),
            }
            for r in rows[:MAX_ITEMS]
        ]

    def get_attendance(self, student_email: str) -> list[dict[str, Any]]:
        out = []
        for c in self._provider.attendance(student_email)[:MAX_ITEMS]:
            pct = float(c.get("attendancePercentage") or 0)
            total = int(c.get("totalClasses") or 0)
            out.append(
                {
                    "course": c.get("courseName"),
                    "attendancePercentage": round(pct, 1),
                    "attendedClasses": c.get("attendedClasses"),
                    "totalClasses": total,
                    "status": attendance_status(pct, total),
                }
            )
        return out
