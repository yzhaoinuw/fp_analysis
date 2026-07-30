"""Derive stable top-level folder names for full Windows packages."""

from __future__ import annotations

import argparse
import re


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


def get_full_zip_name(version: str) -> str:
    return f"{get_package_folder_name(version)}_full.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-zip-name",
        action="store_true",
        help="print the stable major/minor full-package ZIP name",
    )
    parser.add_argument("version")
    args = parser.parse_args()
    if args.full_zip_name:
        print(get_full_zip_name(args.version))
    else:
        print(get_package_folder_name(args.version))
