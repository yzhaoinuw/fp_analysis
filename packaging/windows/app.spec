# -*- mode: python ; coding: utf-8 -*-

from importlib.util import find_spec
import os
from pathlib import Path
import sys


ROOT = Path(os.environ.get("FP_ANALYSIS_REPO_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging" / "windows"))

from fp_analysis_app import VERSION  # noqa: E402
from package_folder_name import get_package_folder_name  # noqa: E402


def package_dir(package_name):
    spec = find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            f"Could not locate package {package_name!r} in the build environment."
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def conda_runtime_binaries(names):
    binary_dir = Path(sys.prefix) / "Library" / "bin"
    binaries = []
    for name in names:
        path = binary_dir / name
        if not path.is_file():
            raise RuntimeError(f"Could not locate required Conda runtime DLL: {path}")
        binaries.append((str(path), "."))
    return binaries


datas = [
    (str(package_dir("dash_extensions")), "dash_extensions"),
    (str(package_dir("scipy")), "scipy"),
]

binaries = conda_runtime_binaries(
    (
        "ffi-8.dll",
        "libbz2.dll",
        "libcrypto-3-x64.dll",
        "libexpat.dll",
        "liblzma.dll",
        "libssl-3-x64.dll",
        "sqlite3.dll",
    )
)

a = Analysis(
    [str(ROOT / "run_desktop_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "desktop_app_source_updater",
        "startup_update_config",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Keep fp_analysis_app patchable beside the executable instead of freezing it
# into _internal.
a.pure = [
    item
    for item in a.pure
    if not (
        item[0] == "fp_analysis_app"
        or item[0].startswith("fp_analysis_app.")
    )
]
a.scripts = [
    item
    for item in a.scripts
    if not (
        item[0] == "fp_analysis_app"
        or item[0].startswith("fp_analysis_app.")
    )
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="run_fp_analysis_app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=get_package_folder_name(VERSION),
)
