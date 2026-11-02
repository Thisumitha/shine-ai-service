from dataclasses import dataclass

from app.config import Settings, get_settings


@dataclass(frozen=True)
class Container:
    """Wires every long-lived object once. Later phases add the embedder, LLM client, stores, etc."""

    settings: Settings


def build_container() -> Container:
    return Container(settings=get_settings())
