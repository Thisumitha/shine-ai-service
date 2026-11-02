from abc import ABC, abstractmethod

from app.domain.entities.user import CurrentUser


class CourseAccess(ABC):
    """Decides which courses a user may read. The server decides, never the browser."""

    @abstractmethod
    def allowed_course_ids(self, user: CurrentUser, token: str) -> set[int]: ...
