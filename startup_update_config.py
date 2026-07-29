from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from desktop_app_source_updater import (
    DEFAULT_BLOCKED_PATH_PREFIXES,
    DEFAULT_BLOCKED_PATH_SUFFIXES,
    UpdateConfig,
)


LATEST_RELEASE_URL = "https://github.com/yzhaoinuw/fp_analysis/releases/latest"
UPDATE_ASSET_PREFIX = "fp_analysis_app_update_"
ALLOWED_PAYLOAD_PATHS = ("fp_analysis_app/",)

BLOCKED_PATH_PREFIXES = (
    *DEFAULT_BLOCKED_PATH_PREFIXES,
    "fp_analysis_app/assets/figures/",
    "fp_analysis_app/assets/spreadsheets/",
    "fp_analysis_app/assets/videos/",
)
BLOCKED_PATH_SUFFIXES = (
    *DEFAULT_BLOCKED_PATH_SUFFIXES,
    ".h5",
    ".mat",
    ".npy",
    ".npz",
    ".pptx",
    ".xls",
    ".xlsx",
)


def _default_check_state_file() -> Path:
    user_state_root = Path(
        os.environ.get("LOCALAPPDATA") or (Path.home() / ".cache")
    )
    return user_state_root / "fp_analysis" / "update-check.json"


def build_startup_update_config(
    app_root: str | os.PathLike[str],
    *,
    update_url: str | None = None,
    latest_release_url: str = LATEST_RELEASE_URL,
    release_api_url: str = "",
    check_state_file: str | os.PathLike[str] | None = None,
    on_update_available: Callable[[str, str], None] | None = None,
) -> UpdateConfig:
    return UpdateConfig(
        app_name="fp_analysis",
        app_root=app_root,
        installed_version_file="fp_analysis_app/__init__.py",
        latest_release_url=latest_release_url,
        latest_release_env="FP_ANALYSIS_UPDATE_LATEST_RELEASE_URL",
        release_api_url=release_api_url,
        asset_prefix=UPDATE_ASSET_PREFIX,
        allowed_payload_paths=ALLOWED_PAYLOAD_PATHS,
        check_state_file=check_state_file or _default_check_state_file(),
        update_url=update_url,
        skip_update_env="FP_ANALYSIS_SKIP_UPDATE",
        update_zip_url_env="FP_ANALYSIS_UPDATE_ZIP_URL",
        release_api_env="FP_ANALYSIS_UPDATE_RELEASE_API",
        asset_prefix_env="FP_ANALYSIS_UPDATE_ASSET_PREFIX",
        timeout_env="FP_ANALYSIS_UPDATE_TIMEOUT_SECONDS",
        force_check_env="FP_ANALYSIS_FORCE_UPDATE_CHECK",
        on_update_available=on_update_available,
        blocked_path_prefixes=BLOCKED_PATH_PREFIXES,
        blocked_path_suffixes=BLOCKED_PATH_SUFFIXES,
    )
