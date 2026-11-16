from abc import ABC, abstractmethod
from typing import Any


class StudentDataError(Exception):
    """Raised when student data cannot be fetched from the upstream system."""


class StudentDataProvider(ABC):
    """Abstract port for read-only student data (owned by the Spring backend).

    Every call is scoped to one student email that the caller has already verified.
    """

    @abstractmethod
    def timetable(self, student_email: str) -> list[dict[str, Any]]:
        """Return the student's calendar slots (classes and exams)."""
        ...

    @abstractmethod
    def invoices(self, student_email: str) -> list[dict[str, Any]]:
        """Return the student's invoices."""
        ...

    @abstractmethod
    def attendance(self, student_email: str) -> list[dict[str, Any]]:
        """Return per-course attendance statistics."""
        ...
