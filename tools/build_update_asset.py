from __future__ import annotations

import sys

from desktop_app_source_updater.build_update_asset import main as build_shared_update_asset


APP_ARGUMENTS = (
    "--app-name",
    "fp_analysis",
    "--runtime-path",
    "fp_analysis_app",
    "--version-file",
    "fp_analysis_app/__init__.py",
    "--asset-prefix",
    "fp_analysis_app_update_",
    "--blocked-path-prefix",
    ".worktrees/",
    "--blocked-path-prefix",
    "archive/",
    "--blocked-path-prefix",
    "build/",
    "--blocked-path-prefix",
    "cache/",
    "--blocked-path-prefix",
    "data/",
    "--blocked-path-prefix",
    "dist/",
    "--blocked-path-prefix",
    "fp_analysis_app/assets/figures/",
    "--blocked-path-prefix",
    "fp_analysis_app/assets/spreadsheets/",
    "--blocked-path-prefix",
    "fp_analysis_app/assets/videos/",
    "--blocked-path-suffix",
    ".lock",
    "--blocked-path-suffix",
    ".spec",
    "--blocked-path-suffix",
    ".h5",
    "--blocked-path-suffix",
    ".mat",
    "--blocked-path-suffix",
    ".npy",
    "--blocked-path-suffix",
    ".npz",
    "--blocked-path-suffix",
    ".pptx",
    "--blocked-path-suffix",
    ".xls",
    "--blocked-path-suffix",
    ".xlsx",
)


def main(argv: list[str] | None = None) -> int:
    forwarded_arguments = sys.argv[1:] if argv is None else argv
    return build_shared_update_asset([*APP_ARGUMENTS, *forwarded_arguments])


if __name__ == "__main__":
    raise SystemExit(main())
