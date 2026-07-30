import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "packaging" / "windows" / "export_runtime_from_git.py"
PACKAGE_FOLDER_NAME_SCRIPT = (
    REPO_ROOT / "packaging" / "windows" / "package_folder_name.py"
)


class TestPackageFolderName(unittest.TestCase):
    def test_uses_major_minor_release_line_for_stable_folder_name(self):
        self.assertEqual(
            "fp_analysis_app_v0.6",
            self._get_package_folder_name("v0.6.0"),
        )
        self.assertEqual(
            "fp_analysis_app_v0.6",
            self._get_package_folder_name("v0.6.9"),
        )
        self.assertEqual(
            "fp_analysis_app_v0.6",
            self._get_package_folder_name("v0.6.0-dev2"),
        )
        self.assertEqual(
            "fp_analysis_app_v0.3",
            self._get_package_folder_name("v0.3-dev"),
        )

    def test_rejects_version_without_major_minor_release_line(self):
        result = subprocess.run(
            [sys.executable, str(PACKAGE_FOLDER_NAME_SCRIPT), "not-a-version"],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("Could not determine", result.stderr)

    def test_full_zip_uses_stable_release_line_and_full_suffix(self):
        self.assertEqual(
            "fp_analysis_app_v0.6_full.zip",
            self._get_full_zip_name("v0.6.0"),
        )
        self.assertEqual(
            "fp_analysis_app_v0.6_full.zip",
            self._get_full_zip_name("v0.6.9"),
        )
        self.assertEqual(
            "fp_analysis_app_v0.6_full.zip",
            self._get_full_zip_name("v0.6.0-dev2"),
        )

    def _get_package_folder_name(self, version):
        return subprocess.run(
            [sys.executable, str(PACKAGE_FOLDER_NAME_SCRIPT), version],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

    def _get_full_zip_name(self, version):
        return subprocess.run(
            [
                sys.executable,
                str(PACKAGE_FOLDER_NAME_SCRIPT),
                "--full-zip-name",
                version,
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()


class TestRuntimeExport(unittest.TestCase):
    def test_clean_export_uses_exact_git_blob_bytes(self):
        with TemporaryDirectory() as temp_dir:
            repo = self._make_repo(Path(temp_dir))
            tracked_bytes = b"VALUE = 'tracked'\r\n"
            self._write_bytes(repo, "fp_analysis_app/app_dev.py", tracked_bytes)
            self._commit(repo, "baseline")
            committed_bytes = self._git_bytes(
                repo,
                "show",
                "HEAD:fp_analysis_app/app_dev.py",
            )
            self._write_bytes(
                repo,
                "fp_analysis_app/untracked.py",
                b"UNTRACKED = True\n",
            )
            destination = Path(temp_dir) / "export"

            self._run_export(
                repo,
                destination,
                "--ref",
                "HEAD",
            )

            self.assertEqual(
                committed_bytes,
                (destination / "fp_analysis_app" / "app_dev.py").read_bytes(),
            )
            self.assertFalse(
                (destination / "fp_analysis_app" / "untracked.py").exists()
            )

    def test_dirty_export_uses_tracked_worktree_bytes(self):
        with TemporaryDirectory() as temp_dir:
            repo = self._make_repo(Path(temp_dir))
            self._write_bytes(
                repo,
                "fp_analysis_app/app_dev.py",
                b"VALUE = 'baseline'\n",
            )
            self._commit(repo, "baseline")
            dirty_bytes = b"VALUE = 'dirty test build'\r\n"
            self._write_bytes(repo, "fp_analysis_app/app_dev.py", dirty_bytes)
            destination = Path(temp_dir) / "export"

            self._run_export(repo, destination, "--worktree")

            self.assertEqual(
                dirty_bytes,
                (destination / "fp_analysis_app" / "app_dev.py").read_bytes(),
            )

    def _make_repo(self, root):
        repo = root / "repo"
        repo.mkdir()
        self._git(repo, "init", "-b", "main")
        self._git(repo, "config", "user.email", "test@example.com")
        self._git(repo, "config", "user.name", "Test User")
        return repo

    def _write_bytes(self, repo, relative_path, data):
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _commit(self, repo, message):
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-m", message)

    def _git(self, repo, *args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=True,
            text=True,
        )

    def _git_bytes(self, repo, *args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=True,
        ).stdout

    def _run_export(self, repo, destination, *mode):
        subprocess.run(
            [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--repo",
                str(repo),
                "--runtime-path",
                "fp_analysis_app",
                "--destination",
                str(destination),
                *mode,
            ],
            capture_output=True,
            check=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
