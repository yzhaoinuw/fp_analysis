# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import collect_data_files

sys.setrecursionlimit(sys.getrecursionlimit() * 5)
# Add the current working directory to sys.path
sys.path.insert(0, os.getcwd())

from fp_analysis_app import VERSION


datas=[
    ('C:\\Users\\yzhao\\miniconda3\\envs\\fiber_photometry\\lib\\site-packages\\dash_extensions', 'dash_extensions'),
    ('C:\\Users\\yzhao\\python_projects\\fp_analysis\\fp_analysis_app\\assets', 'assets'),
    ('C:\\Users\\yzhao\\miniconda3\\envs\\fiber_photometry\\lib\\site-packages\\scipy', 'scipy'),
]

a = Analysis(
    ['main.py'],
    pathex=['C:\\Users\\yzhao\\miniconda3\\envs\\fiber_photometry\\lib\\site-packages'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Remove fp_analysis_app modules from the Analysis

a.pure = [x for x in a.pure if 'fp_analysis_app' not in x[0]]
a.scripts = [x for x in a.scripts if 'fp_analysis_app' not in x[0]]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run_fp_analysis_app',
    debug=True,
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
    name=f'fp_analysis_app_{VERSION}',
)
