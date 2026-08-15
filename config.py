"""
Configuration settings for the hiring agent application.
"""

import os


def _is_true(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


# Global development mode flag.
# Defaults to on locally and off on Vercel (serverless filesystems are
# read-only except /tmp, so local cache/CSV writes would fail there).
# Can be overridden explicitly with the DEVELOPMENT_MODE environment variable.
DEVELOPMENT_MODE = _is_true(
    os.environ.get(
        "DEVELOPMENT_MODE",
        "0" if os.environ.get("VERCEL") == "1" else "1",
    )
)
