from __future__ import annotations

import os

from desktop_app_source_updater import (
    DEFAULT_BLOCKED_PATH_PREFIXES,
    DEFAULT_BLOCKED_PATH_SUFFIXES,
    UpdateConfig,
)


RELEASE_API_URL = "https://api.github.com/repos/yzhaoinuw/fp_analysis/releases/latest"
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


def build_startup_update_config(
    app_root: str | os.PathLike[str],
    *,
    update_url: str | None = None,
    release_api_url: str = RELEASE_API_URL,
) -> UpdateConfig:
    return UpdateConfig(
        app_name="fp_analysis",
        app_root=app_root,
        installed_version_file="fp_analysis_app/__init__.py",
        release_api_url=release_api_url,
        asset_prefix=UPDATE_ASSET_PREFIX,
        allowed_payload_paths=ALLOWED_PAYLOAD_PATHS,
        update_url=update_url,
        skip_update_env="FP_ANALYSIS_SKIP_UPDATE",
        update_zip_url_env="FP_ANALYSIS_UPDATE_ZIP_URL",
        release_api_env="FP_ANALYSIS_UPDATE_RELEASE_API",
        asset_prefix_env="FP_ANALYSIS_UPDATE_ASSET_PREFIX",
        timeout_env="FP_ANALYSIS_UPDATE_TIMEOUT_SECONDS",
        blocked_path_prefixes=BLOCKED_PATH_PREFIXES,
        blocked_path_suffixes=BLOCKED_PATH_SUFFIXES,
    )
