import logging

# Libraries that log every HTTP call at INFO level ("HTTP Request: POST ... 200 OK", "AFC is enabled ...").
_NOISY = ("httpx", "httpx2", "httpcore", "httpcore2", "google_genai", "google.genai", "chromadb", "urllib3")


def quiet_libraries() -> None:
    """Show warnings and errors only, so the CLI output stays readable.

    Set SHINE_DEBUG=1 to see everything again.
    """
    import os

    if os.environ.get("SHINE_DEBUG"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        return
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
