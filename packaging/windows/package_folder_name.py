"""Derive stable top-level folder names for full Windows packages."""

from __future__ import annotations

import re
import sys


PACKAGE_FOLDER_PREFIX = "fp_analysis_app"
RELEASE_LINE_PATTERN = re.compile(
    r"^(v?\d+)\.(\d+)(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?$"
)


def get_release_line(version: str) -> str:
    version = str(version).strip()
    match = RELEASE_LINE_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(
            f"Could not determine the major/minor release line from {version!r}."
        )
    return f"{match.group(1)}.{match.group(2)}"


def get_package_folder_name(version: str) -> str:
    return f"{PACKAGE_FOLDER_PREFIX}_{get_release_line(version)}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: package_folder_name.py <version>")
    print(get_package_folder_name(sys.argv[1]))
