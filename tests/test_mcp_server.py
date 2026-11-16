import asyncio
import json

from mcp.server.fastmcp import FastMCP

from app.mcp_server.server import build_mcp_server, current_student_email
from app.services.student_tools import StudentToolService
from tests.services.test_student_data import FakeProvider


def _server(provider: FakeProvider) -> FastMCP:
    return build_mcp_server(StudentToolService(provider))


def test_tool_schemas_have_no_email_argument():
    tools = asyncio.run(_server(FakeProvider()).list_tools())
    assert {t.name for t in tools} == {"get_timetable", "get_invoices", "get_attendance"}
    for tool in tools:
        assert "email" not in json.dumps(tool.inputSchema).lower()


def test_tool_runs_for_context_email():
    provider = FakeProvider()
    token = current_student_email.set("me@x.com")
    try:
        asyncio.run(_server(provider).call_tool("get_attendance", {}))
    finally:
        current_student_email.reset(token)
    assert provider.emails == ["me@x.com"]


def test_tool_refuses_without_verified_student(monkeypatch):
    monkeypatch.delenv("MCP_STUDENT_EMAIL", raising=False)
    provider = FakeProvider()
    try:
        asyncio.run(_server(provider).call_tool("get_invoices", {}))
    except Exception:
        pass
    assert provider.emails == []
