import hashlib
import json
import shutil
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_app_source_updater import run_startup_update
from startup_update_config import build_startup_update_config


def sha256(data):
    return hashlib.sha256(data).hexdigest()


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseZipFixture:
    def __init__(self, temp_dir):
        self.root = Path(temp_dir)
        self.app_root = self.root / "app"
        self.release_dir = self.root / "release"
        self.release_dir.mkdir()

    def setup_installed_app(self, version="v0.5.0"):
        self.write_app_file("fp_analysis_app/__init__.py", f'VERSION = "{version}"\n')
        self.write_app_file("fp_analysis_app/app_dev.py", "APP_VALUE = 'old'\n")

    def config(self, **overrides):
        return build_startup_update_config(self.app_root, **overrides)

    def write_app_file(self, relative_path, text):
        path = self.app_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def read_app_file(self, relative_path):
        return (self.app_root / relative_path).read_text(encoding="utf-8")

    def build_update_zip(
        self,
        *,
        version="v0.5.1",
        payloads=None,
        from_versions=None,
        minimum_version=None,
        include_previous_hashes=True,
        previous_payloads_by_version=None,
        zip_name=None,
    ):
        payloads = payloads or {
            "fp_analysis_app/__init__.py": f'VERSION = "{version}"\n',
            "fp_analysis_app/app_dev.py": "APP_VALUE = 'new'\n",
        }
        files = []
        for relative_path, text in payloads.items():
            entry = {
                "path": relative_path,
                "sha256": sha256(text.encode("utf-8")),
            }
            if include_previous_hashes and previous_payloads_by_version is not None:
                entry["previous_sha256_by_version"] = {
                    version: (
                        None
                        if relative_path not in previous_payloads
                        else sha256(previous_payloads[relative_path].encode("utf-8"))
                    )
                    for version, previous_payloads in previous_payloads_by_version.items()
                }
            elif include_previous_hashes:
                installed_path = self.app_root / relative_path
                if installed_path.exists():
                    versions = from_versions or ["v0.5.0"]
                    entry["previous_sha256_by_version"] = {
                        version: sha256(installed_path.read_bytes())
                        for version in versions
                    }
            files.append(entry)

        manifest = {
            "schema_version": 1,
            "app": "fp_analysis",
            "version": version,
            "changed_files": list(payloads.keys()),
            "files": files,
        }
        if from_versions is not None:
            manifest["from_versions"] = from_versions
        if minimum_version is not None:
            manifest["minimum_version"] = minimum_version

        update_zip = self.release_dir / (zip_name or f"fp_analysis_app_update_{version}.zip")
        with zipfile.ZipFile(update_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for relative_path, text in payloads.items():
                zf.writestr(relative_path, text)
        return update_zip

    def build_release_metadata(self, asset_path, tag_name="v0.5.1"):
        metadata_path = self.release_dir / "latest_release.json"
        metadata = {
            "tag_name": tag_name,
            "assets": [
                {
                    "name": asset_path.name,
                    "browser_download_url": str(asset_path),
                }
            ],
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return metadata_path


class TestStartupUpdate(unittest.TestCase):
    def test_uses_fp_analysis_shared_updater_contract(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)

            config = fixture.config()

            self.assertEqual("fp_analysis", config.app_name)
            self.assertEqual("fp_analysis_app/__init__.py", config.installed_version_file)
            self.assertEqual(("fp_analysis_app/",), config.allowed_payload_paths)
            self.assertEqual("fp_analysis_app_update_", config.asset_prefix)
            self.assertIn("fp_analysis_app/assets/videos/", config.blocked_path_prefixes)
            self.assertIn(".mat", config.blocked_path_suffixes)

    def test_applies_compatible_release_zip(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app()
            update_zip = fixture.build_update_zip(from_versions=["v0.5.0"])

            result = run_startup_update(fixture.config(update_url=str(update_zip)))

            self.assertEqual("updated", result.status)
            self.assertEqual(
                (
                    "fp_analysis_app/__init__.py",
                    "fp_analysis_app/app_dev.py",
                ),
                result.changed_files,
            )
            self.assertEqual('VERSION = "v0.5.1"\n', fixture.read_app_file("fp_analysis_app/__init__.py"))
            self.assertEqual("APP_VALUE = 'new'\n", fixture.read_app_file("fp_analysis_app/app_dev.py"))

    def test_jumps_from_any_supported_older_version(self):
        previous_payloads_by_version = {
            "v0.5.0": {
                "fp_analysis_app/__init__.py": 'VERSION = "v0.5.0"\n',
                "fp_analysis_app/app_dev.py": "APP_VALUE = 'old0'\n",
            },
            "v0.5.1": {
                "fp_analysis_app/__init__.py": 'VERSION = "v0.5.1"\n',
                "fp_analysis_app/app_dev.py": "APP_VALUE = 'old1'\n",
            },
            "v0.5.2": {
                "fp_analysis_app/__init__.py": 'VERSION = "v0.5.2"\n',
                "fp_analysis_app/app_dev.py": "APP_VALUE = 'old2'\n",
            },
        }

        for installed_version, installed_payloads in previous_payloads_by_version.items():
            with self.subTest(installed_version=installed_version):
                with TemporaryDirectory() as temp_dir:
                    fixture = ReleaseZipFixture(temp_dir)
                    for relative_path, text in installed_payloads.items():
                        fixture.write_app_file(relative_path, text)
                    update_zip = fixture.build_update_zip(
                        version="v0.5.3",
                        from_versions=list(previous_payloads_by_version),
                        previous_payloads_by_version=previous_payloads_by_version,
                        payloads={
                            "fp_analysis_app/__init__.py": 'VERSION = "v0.5.3"\n',
                            "fp_analysis_app/app_dev.py": "APP_VALUE = 'new3'\n",
                        },
                    )

                    result = run_startup_update(fixture.config(update_url=str(update_zip)))

                    self.assertEqual("updated", result.status)
                    self.assertEqual(
                        'VERSION = "v0.5.3"\n',
                        fixture.read_app_file("fp_analysis_app/__init__.py"),
                    )
                    self.assertEqual(
                        "APP_VALUE = 'new3'\n",
                        fixture.read_app_file("fp_analysis_app/app_dev.py"),
                    )

    def test_blocks_unsupported_older_version(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app(version="v0.4.9")
            update_zip = fixture.build_update_zip(
                version="v0.5.3",
                from_versions=["v0.5.0", "v0.5.1", "v0.5.2"],
            )

            result = run_startup_update(fixture.config(update_url=str(update_zip)))

            self.assertEqual("blocked", result.status)
            self.assertIn("not compatible", result.message)
            self.assertEqual('VERSION = "v0.4.9"\n', fixture.read_app_file("fp_analysis_app/__init__.py"))

    def test_discovers_update_zip_from_release_metadata(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app()
            update_zip = fixture.build_update_zip(from_versions=["v0.5.0"])
            release_metadata = fixture.build_release_metadata(update_zip)

            result = run_startup_update(
                fixture.config(release_api_url=str(release_metadata))
            )

            self.assertEqual("updated", result.status)
            self.assertEqual('VERSION = "v0.5.1"\n', fixture.read_app_file("fp_analysis_app/__init__.py"))

    def test_skips_same_version_release_zip(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app()
            update_zip = fixture.build_update_zip(version="v0.5.0")

            result = run_startup_update(fixture.config(update_url=str(update_zip)))

            self.assertEqual("up-to-date", result.status)
            self.assertEqual("APP_VALUE = 'old'\n", fixture.read_app_file("fp_analysis_app/app_dev.py"))


class TestBuildUpdateAsset(unittest.TestCase):
    def test_builds_multi_baseline_update_asset(self):
        if shutil.which("git") is None:
            self.skipTest("git is not available on PATH")

        with TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "repo"
            repo.mkdir()
            self._git(repo, "init", "-b", "main")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test User")

            self._write(repo, "fp_analysis_app/__init__.py", 'VERSION = "v0.5.0"\n')
            self._write(repo, "fp_analysis_app/app_dev.py", "APP_VALUE = 'old0'\n")
            self._commit(repo, "v0.5.0")
            self._git(repo, "tag", "v0.5.0")

            self._write(repo, "fp_analysis_app/__init__.py", 'VERSION = "v0.5.1"\n')
            self._write(repo, "fp_analysis_app/app_dev.py", "APP_VALUE = 'old1'\n")
            self._commit(repo, "v0.5.1")
            self._git(repo, "tag", "v0.5.1")

            self._write(repo, "fp_analysis_app/__init__.py", 'VERSION = "v0.5.3"\n')
            self._write(repo, "fp_analysis_app/app_dev.py", "APP_VALUE = 'new3'\n")
            self._commit(repo, "v0.5.3")

            output_zip = Path(temp_dir) / "fp_analysis_app_update_v0.5.3.zip"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "build_update_asset.py"),
                    "--repo",
                    str(repo),
                    "--from-ref",
                    "v0.5.0",
                    "--from-ref",
                    "v0.5.1",
                    "--to-ref",
                    "HEAD",
                    "--output",
                    str(output_zip),
                ],
                capture_output=True,
                check=True,
                text=True,
            )

            with zipfile.ZipFile(output_zip) as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

            self.assertEqual(["v0.5.0", "v0.5.1"], manifest["from_versions"])
            app_dev_entry = next(
                item
                for item in manifest["files"]
                if item["path"] == "fp_analysis_app/app_dev.py"
            )
            self.assertEqual(
                {"v0.5.0", "v0.5.1"},
                set(app_dev_entry["previous_sha256_by_version"]),
            )

    def test_builder_refuses_dependency_changes(self):
        if shutil.which("git") is None:
            self.skipTest("git is not available on PATH")

        with TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "repo"
            repo.mkdir()
            self._git(repo, "init", "-b", "main")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test User")

            self._write(repo, "fp_analysis_app/__init__.py", 'VERSION = "v0.5.0"\n')
            self._write(repo, "fp_analysis_app/app_dev.py", "APP_VALUE = 'old'\n")
            self._write(repo, "requirements.txt", "dash==2\n")
            self._commit(repo, "v0.5.0")
            self._git(repo, "tag", "v0.5.0")

            self._write(repo, "fp_analysis_app/__init__.py", 'VERSION = "v0.5.1"\n')
            self._write(repo, "fp_analysis_app/app_dev.py", "APP_VALUE = 'new'\n")
            self._write(repo, "requirements.txt", "dash==3\n")
            self._commit(repo, "v0.5.1")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "build_update_asset.py"),
                    "--repo",
                    str(repo),
                    "--from-ref",
                    "v0.5.0",
                    "--to-ref",
                    "HEAD",
                ],
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertEqual(1, result.returncode)
            self.assertIn("requirements.txt", result.stderr)

    def _write(self, repo, relative_path, text):
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

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

    def test_runtime_blocks_generated_data_paths(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app()
            update_zip = fixture.build_update_zip(
                payloads={
                    "fp_analysis_app/__init__.py": 'VERSION = "v0.5.1"\n',
                    "fp_analysis_app/assets/videos/output.mat": "generated data\n",
                },
                from_versions=["v0.5.0"],
            )

            result = run_startup_update(fixture.config(update_url=str(update_zip)))

            self.assertEqual("blocked", result.status)
            self.assertIn("packaged refresh required", result.message)
            self.assertEqual('VERSION = "v0.5.0"\n', fixture.read_app_file("fp_analysis_app/__init__.py"))

    def test_skips_when_local_file_hash_differs_from_manifest_baseline(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app()
            update_zip = fixture.build_update_zip(from_versions=["v0.5.0"])
            fixture.write_app_file("fp_analysis_app/app_dev.py", "APP_VALUE = 'local edit'\n")

            result = run_startup_update(fixture.config(update_url=str(update_zip)))

            self.assertEqual("skipped", result.status)
            self.assertIn("differ from the update baseline", result.message)
            self.assertEqual(
                "APP_VALUE = 'local edit'\n",
                fixture.read_app_file("fp_analysis_app/app_dev.py"),
            )

    def test_skips_when_manifest_lacks_previous_hash_for_existing_file(self):
        with TemporaryDirectory() as temp_dir:
            fixture = ReleaseZipFixture(temp_dir)
            fixture.setup_installed_app()
            update_zip = fixture.build_update_zip(
                include_previous_hashes=False,
                from_versions=["v0.5.0"],
            )

            result = run_startup_update(fixture.config(update_url=str(update_zip)))

            self.assertEqual("skipped", result.status)
            self.assertIn("cannot verify local source state", result.message)
            self.assertEqual("APP_VALUE = 'old'\n", fixture.read_app_file("fp_analysis_app/app_dev.py"))


if __name__ == "__main__":
    unittest.main()
