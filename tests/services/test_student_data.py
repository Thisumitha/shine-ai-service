import json
from datetime import date, timedelta
from unittest.mock import MagicMock

import httpx
import pytest

from app.domain.ports.llm_client import LLMClient
from app.domain.ports.student_data import StudentDataError, StudentDataProvider
from app.infrastructure.clients.spring_student_data import SpringStudentDataClient
from app.services.agent.student_data import (
    NO_STUDENT_REPLY,
    UNAVAILABLE_REPLY,
    StudentDataStage,
)
from app.services.student_tools import StudentToolService, attendance_status

TODAY = date.today().isoformat()
TOMORROW = (date.today() + timedelta(days=1)).isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


class FakeProvider(StudentDataProvider):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.emails: list[str] = []

    def _guard(self, email: str) -> None:
        self.emails.append(email)
        if self.fail:
            raise StudentDataError("down")

    def timetable(self, student_email):
        self._guard(student_email)
        return [
            {"className": "Maths", "eventType": "CLASS", "date": YESTERDAY, "startTime": "08:00", "endTime": "09:00", "status": "COMPLETED"},
            {"className": "Physics", "eventType": "CLASS", "date": TOMORROW, "startTime": "10:00", "endTime": "11:00", "status": "PENDING"},
            {"className": "Chem", "eventType": "CLASS", "date": TODAY, "startTime": "14:00", "endTime": "15:00", "status": "PENDING"},
        ]

    def invoices(self, student_email):
        self._guard(student_email)
        return [
            {"invoiceNumber": "INV-1", "totalAmount": 5000, "paymentStatus": "PENDING", "dueDate": "2026-11-15",
             "paymentLink": "https://pay.example/secret", "bankSlipUrl": "slip.png", "items": "[]"},
            {"invoiceNumber": "INV-2", "totalAmount": 3000, "paymentStatus": "PAID", "paidAt": "2026-10-01"},
        ]

    def attendance(self, student_email):
        self._guard(student_email)
        return [
            {"courseName": "Maths", "attendancePercentage": 45.0, "totalClasses": 10, "attendedClasses": 4},
            {"courseName": "Physics", "attendancePercentage": 90.0, "totalClasses": 10, "attendedClasses": 9},
            {"courseName": "Chem", "attendancePercentage": 0, "totalClasses": 0, "attendedClasses": 0},
        ]


def test_timetable_defaults_to_upcoming_sorted():
    names = [s["name"] for s in StudentToolService(FakeProvider()).get_timetable("a@x.com")]
    assert names == ["Chem", "Physics"]


def test_timetable_filters_by_date():
    slots = StudentToolService(FakeProvider()).get_timetable("a@x.com", YESTERDAY)
    assert [s["name"] for s in slots] == ["Maths"]


def test_invoices_hide_payment_links_and_filter_status():
    rows = StudentToolService(FakeProvider()).get_invoices("a@x.com", "pending")
    assert [r["invoiceNumber"] for r in rows] == ["INV-1"]
    assert rows[0]["currency"] == "LKR"
    assert "paymentLink" not in json.dumps(rows) and "slip" not in json.dumps(rows)


def test_attendance_status_thresholds():
    assert attendance_status(45, 10).startswith("Student at risk")
    assert attendance_status(50, 10) == "Consistent Participation Required"
    assert attendance_status(80, 10) == "Consistent Participation Required"
    assert attendance_status(80.1, 10) == "Excellent"
    assert attendance_status(0, 0) == "No Sessions Yet"


def test_attendance_labels():
    rows = StudentToolService(FakeProvider()).get_attendance("a@x.com")
    assert [r["status"] for r in rows] == [
        "Student at risk - Future attendance Compulsory",
        "Excellent",
        "No Sessions Yet",
    ]


def _llm(calls: list[dict], reply: str = "ok") -> MagicMock:
    llm = MagicMock(spec=LLMClient)
    llm.generate_json.return_value = json.dumps({"calls": calls})
    llm.generate.return_value = reply
    return llm


def test_stage_runs_chosen_tool_for_verified_email_only():
    provider = FakeProvider()
    # The model tries to smuggle in another student's email via args.
    llm = _llm([{"tool": "get_invoices", "args": {"status": "PENDING", "email": "victim@x.com"}}], "You owe Rs. 5000")
    stage = StudentDataStage(llm, StudentToolService(provider))

    assert stage.answer("me@x.com", "any unpaid invoices?") == "You owe Rs. 5000"
    assert provider.emails == ["me@x.com"]
    assert "INV-1" in llm.generate.call_args.kwargs["prompt"]


def test_stage_without_verified_student_does_not_touch_data():
    provider = FakeProvider()
    stage = StudentDataStage(_llm([{"tool": "get_invoices"}]), StudentToolService(provider))
    assert stage.answer(None, "my invoices") == NO_STUDENT_REPLY
    assert provider.emails == []


def test_stage_ignores_unknown_tools():
    provider = FakeProvider()
    stage = StudentDataStage(_llm([{"tool": "delete_everything"}]), StudentToolService(provider))
    stage.answer("me@x.com", "q")
    assert provider.emails == []


def test_stage_returns_friendly_message_when_spring_is_down():
    stage = StudentDataStage(_llm([{"tool": "get_attendance"}]), StudentToolService(FakeProvider(fail=True)))
    assert stage.answer("me@x.com", "my attendance") == UNAVAILABLE_REPLY


def test_stage_survives_bad_tool_selection_json():
    llm = MagicMock(spec=LLMClient)
    llm.generate_json.return_value = "not json"
    llm.generate.return_value = "fallback"
    assert StudentDataStage(llm, StudentToolService(FakeProvider())).answer("me@x.com", "q") == "fallback"


def test_spring_client_sends_secret_and_email():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["secret"] = request.headers.get("X-Internal-Secret")
        seen["email"] = request.url.params.get("studentEmail")
        seen["path"] = request.url.path
        return httpx.Response(200, json=[{"a": 1}])

    client = SpringStudentDataClient("http://spring/", "s3cret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.invoices("me@x.com") == [{"a": 1}]
    assert seen == {"secret": "s3cret", "email": "me@x.com", "path": "/internal/student/invoices"}


def test_spring_client_wraps_http_errors():
    client = SpringStudentDataClient(
        "http://spring", "s", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    )
    with pytest.raises(StudentDataError):
        client.timetable("me@x.com")


def test_act_stage_routes_student_data_intent_with_email():
    from app.services.agent.act import ActStage
    from app.services.agent.reason import INTENT_STUDENT_DATA, ReasonResult

    provider = FakeProvider()
    stage = StudentDataStage(_llm([{"tool": "get_attendance"}], "Maths is at risk"), StudentToolService(provider))
    act = ActStage(MagicMock(), MagicMock(), MagicMock(), student_data_stage=stage)

    answer, citations = act.act(
        user_id="me@x.com",
        course_id=1,
        query="my attendance?",
        reason_result=ReasonResult(INTENT_STUDENT_DATA, "personal", False),
        student_email="me@x.com",
    )

    assert answer == "Maths is at risk" and citations == []
    assert provider.emails == ["me@x.com"]


def test_stage_passes_recent_history_for_follow_ups():
    from app.domain.entities.message import Message

    llm = _llm([{"tool": "get_invoices"}], "The second is LKR 185.25")
    stage = StudentDataStage(llm, StudentToolService(FakeProvider()))
    history = [Message(message_id="1", session_id="s", role="user", content="any pending invoices?")]

    stage.answer("me@x.com", "and the second one?", history)

    assert "any pending invoices?" in llm.generate_json.call_args.kwargs["prompt"]
    assert "any pending invoices?" in llm.generate.call_args.kwargs["prompt"]
