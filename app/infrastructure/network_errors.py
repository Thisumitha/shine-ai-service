"""Recognise a temporary network failure (DNS, refused or dropped connection, timeout).

The Google SDK raises errors from whichever HTTP library it bundles (`httpx`, or `httpx2` in newer
releases), so checking for one library's classes misses the others. Anything raised by these libraries,
and any socket-level error, is treated as a network failure that is worth retrying.
"""

_HTTP_LIBRARIES = {"httpx", "httpx2", "httpcore", "httpcore2"}


def is_network_error(err: BaseException) -> bool:
    if isinstance(err, (OSError, TimeoutError)):  # socket.gaierror ("getaddrinfo failed") is an OSError
        return True
    return any(cls.__module__.split(".")[0] in _HTTP_LIBRARIES for cls in type(err).__mro__)
