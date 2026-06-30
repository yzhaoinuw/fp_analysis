from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


RUNTIME_PATHS = ("fp_analysis_app", "startup_update.py")
DEPENDENCY_OR_BUILD_FILES = {
    "app.spec",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}
DEPENDENCY_OR_BUILD_SUFFIXES = {".lock", ".spec"}
LOCAL_ARTIFACT_PREFIXES = (
    ".worktrees/",
    "archive/",
    "build/",
    "cache/",
    "data/",
    "dist/",
    "fp_analysis_app/assets/figures/",
    "fp_analysis_app/assets/spreadsheets/",
    "fp_analysis_app/assets/videos/",
)
LOCAL_ARTIFACT_SUFFIXES = (
    ".h5",
    ".mat",
    ".npy",
    ".npz",
    ".pptx",
    ".xls",
    ".xlsx",
)


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    version = args.version or read_version_from_ref(repo, args.to_ref)
    from_refs = args.from_refs
    from_versions_by_ref = {
        from_ref: read_version_from_ref(repo, from_ref)
        for from_ref in from_refs
    }
    output = args.output or repo / "dist" / f"fp_analysis_app_update_{version}.zip"

    all_changed = sorted(
        {
            path
            for from_ref in from_refs
            for path in git_lines(
                repo,
                "diff",
                "--name-only",
                f"{from_ref}..{args.to_ref}",
            )
        }
    )
    blocked = [path for path in all_changed if requires_packaged_refresh(path)]
    if blocked:
        print(
            "Refusing to build a source-only update because these paths changed:",
            file=sys.stderr,
        )
        for path in blocked:
            print(f"  {path}", file=sys.stderr)
        return 1

    changed_runtime = sorted(
        {
            path
            for from_ref in from_refs
            for path in changed_runtime_paths(repo, from_ref, args.to_ref)
        }
    )
    if not changed_runtime:
        print("No runtime source files changed.", file=sys.stderr)
        return 1

    manifest_files = []
    payloads = {}
    for path in changed_runtime:
        current_bytes = git_file_bytes(repo, args.to_ref, path)
        previous_sha256_by_version = {}
        for from_ref, from_version in from_versions_by_ref.items():
            previous_bytes = git_file_bytes(repo, from_ref, path, allow_missing=True)
            previous_sha256_by_version[from_version] = (
                None if previous_bytes is None else sha256(previous_bytes)
            )
        item = {
            "path": path,
            "sha256": sha256(current_bytes),
            "previous_sha256_by_version": previous_sha256_by_version,
        }
        manifest_files.append(item)
        payloads[path] = current_bytes

    manifest = {
        "schema_version": 1,
        "app": "fp_analysis",
        "version": version,
        "from_versions": list(from_versions_by_ref.values()),
        "changed_files": changed_runtime,
        "files": manifest_files,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for path, data in payloads.items():
            zf.writestr(path, data)

    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a code-only fp_analysis update zip for GitHub Releases."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--from-ref",
        action="append",
        dest="from_refs",
        required=True,
        help=(
            "Previous release tag or commit users may have installed. "
            "Repeat to support skipped-release jump-ahead updates."
        ),
    )
    parser.add_argument(
        "--to-ref",
        default="HEAD",
        help="New release tag or commit to package. Defaults to HEAD.",
    )
    parser.add_argument(
        "--version",
        help="Version string for the update manifest. Defaults to VERSION in to-ref.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output zip path. Defaults to dist/fp_analysis_app_update_<version>.zip.",
    )
    return parser.parse_args()


def changed_runtime_paths(repo: Path, from_ref: str, to_ref: str) -> list[str]:
    result = git_lines(
        repo,
        "diff",
        "--name-status",
        f"{from_ref}..{to_ref}",
        "--",
        *RUNTIME_PATHS,
    )
    paths = []
    unsupported = []
    for line in result:
        parts = line.split("\t")
        status = parts[0]
        if status.startswith(("A", "M")) and len(parts) == 2:
            path = normalize_path(parts[1])
            if is_allowed_payload_path(path):
                paths.append(path)
        else:
            unsupported.append(line)

    if unsupported:
        print(
            "Runtime deletions, renames, or complex changes need a packaged refresh:",
            file=sys.stderr,
        )
        for line in unsupported:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)

    return sorted(paths)


def read_version_from_ref(repo: Path, ref: str) -> str:
    text = git_file_bytes(repo, ref, "fp_analysis_app/__init__.py").decode("utf-8")
    match = re.search(r"VERSION\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise SystemExit(f"Could not read VERSION from {ref}:fp_analysis_app/__init__.py")
    return match.group(1)


def git_lines(repo: Path, *args: str) -> list[str]:
    result = run_git(repo, *args)
    return [line for line in result.stdout.splitlines() if line]


def git_file_bytes(
    repo: Path,
    ref: str,
    path: str,
    *,
    allow_missing: bool = False,
) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout
    if allow_missing:
        return None
    message = result.stderr.decode("utf-8", errors="replace").strip()
    raise SystemExit(message or f"Could not read {ref}:{path}")


def run_git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    if check and result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git command failed")
    return result


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def is_allowed_payload_path(path: str) -> bool:
    return path == "startup_update.py" or path.startswith("fp_analysis_app/")


def requires_packaged_refresh(path: str) -> bool:
    normalized = normalize_path(path)
    root_name = normalized.rsplit("/", maxsplit=1)[-1]
    suffix = Path(root_name).suffix.lower()
    return (
        normalized in DEPENDENCY_OR_BUILD_FILES
        or root_name in DEPENDENCY_OR_BUILD_FILES
        or suffix in DEPENDENCY_OR_BUILD_SUFFIXES
        or normalized.startswith(LOCAL_ARTIFACT_PREFIXES)
        or suffix in LOCAL_ARTIFACT_SUFFIXES
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
