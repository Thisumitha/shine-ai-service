from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """A reference to a source chunk in course material supporting a generated claim."""

    source_id: str
    file_name: str
    page_number: int
    snippet: str
    course_id: int | None = None
    item_id: int | None = None

