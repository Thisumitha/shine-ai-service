from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, course_ids: set[int], top_k: int) -> list[dict]: ...
