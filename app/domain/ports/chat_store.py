from abc import ABC, abstractmethod


class ChatStore(ABC):
    @abstractmethod
    def save_message(self, session_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    def history(self, session_id: str, limit: int) -> list[dict]: ...
