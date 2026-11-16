"""MCP server exposing the read-only student-data tools.

Run standalone for testing:  python -m app.mcp_server.server
(set MCP_STUDENT_EMAIL to the student to act as, then open it with the MCP Inspector).

The student email is never a tool argument. It comes from `current_student_email`,
which the caller sets after Spring has verified the student.
"""

import json
import os
from contextvars import ContextVar

from mcp.server.fastmcp import FastMCP

from app.services.student_tools import StudentToolService

current_student_email: ContextVar[str | None] = ContextVar("current_student_email", default=None)


def _resolve_email() -> str:
    email = current_student_email.get() or os.environ.get("MCP_STUDENT_EMAIL")
    if not email:
        raise ValueError("No verified student is attached to this request.")
    return email


def build_mcp_server(tools: StudentToolService) -> FastMCP:
    server = FastMCP("shine-student")

    @server.tool()
    def get_timetable(date: str | None = None) -> str:
        """The student's classes and exams. Optional date (YYYY-MM-DD) limits to one day."""
        return json.dumps(tools.get_timetable(_resolve_email(), date), default=str)

    @server.tool()
    def get_invoices(status: str | None = None) -> str:
        """The student's payment invoices. Optional status: PENDING or PAID."""
        return json.dumps(tools.get_invoices(_resolve_email(), status), default=str)

    @server.tool()
    def get_attendance() -> str:
        """The student's attendance percentage per course with a status label."""
        return json.dumps(tools.get_attendance(_resolve_email()), default=str)

    return server


if __name__ == "__main__":
    from app.config import get_settings

    settings = get_settings()
    if settings.spring_base_url:
        from app.infrastructure.clients.spring_student_data import SpringStudentDataClient

        provider = SpringStudentDataClient(settings.spring_base_url, settings.internal_secret)
    else:
        from app.infrastructure.clients.mock_student_data import MockStudentDataClient

        provider = MockStudentDataClient()
        os.environ.setdefault("MCP_STUDENT_EMAIL", "demo.student@shine.lk")
    build_mcp_server(StudentToolService(provider)).run()
