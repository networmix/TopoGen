from __future__ import annotations

import base64
import uuid


def new_base64_uuid() -> str:
    """Return a 22-character URL-safe Base64-encoded UUID without padding.

    The function generates a random version 4 UUID, encodes the 16 raw bytes
    using URL-safe Base64, removes the two trailing padding characters, and
    decodes to ASCII. The resulting string length is 22 characters.

    Returns:
        A 22-character URL-safe Base64 representation of a UUID4 without
        padding.
    """
    return base64.urlsafe_b64encode(uuid.uuid4().bytes)[:-2].decode("ascii")
