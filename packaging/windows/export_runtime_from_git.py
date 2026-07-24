"""Export tracked runtime files without checkout line-ending transformations."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath


def git_output(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(message or "git command failed")
    return result.stdout


def tracked_paths(
    repo: Path,
    runtime_path: str,
    *,
    ref: str | None,
) -> list[str]:
    if ref is None:
        output = git_output(repo, "ls-files", "-z", "--", runtime_path)
    else:
        output = git_output(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            ref,
            "--",
            runtime_path,
        )
    return [path.decode() for path in output.split(b"\0") if path]


def export_runtime(
    repo: Path,
    runtime_path: str,
    destination: Path,
    *,
    ref: str | None,
) -> list[str]:
    paths = tracked_paths(repo, runtime_path, ref=ref)
    if not paths:
        source = "the worktree" if ref is None else ref
        raise ValueError(f"no tracked files found at {source}:{runtime_path}")

    exported = []
    for path in paths:
        relative_path = PurePosixPath(path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe tracked path: {path}")

        if ref is None:
            source_path = repo.joinpath(*relative_path.parts)
            if not source_path.is_file():
                continue
            data = source_path.read_bytes()
        else:
            data = git_output(repo, "show", f"{ref}:{path}")

        output_path = destination.joinpath(*relative_path.parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        exported.append(path)

    if not exported:
        raise ValueError(f"no runtime files were exported for {runtime_path}")
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export tracked runtime files using exact Git or worktree bytes."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--ref",
        help="Git ref to export. Omit with --worktree.",
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Export tracked files from the current worktree for a dirty test build.",
    )
    args = parser.parse_args(argv)

    if bool(args.ref) == args.worktree:
        parser.error("choose exactly one of --ref or --worktree")

    export_runtime(
        args.repo.resolve(),
        args.runtime_path,
        args.destination.resolve(),
        ref=None if args.worktree else args.ref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
